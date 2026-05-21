# -*- coding: utf-8 -*-
"""
MMoE multi-task training for drug-price regression + RDMIT classification.

Expected inputs (from drug_demo_price_reg_rdmit_pytorch.data_preparation())
----------------------------------------------------------------------------
data dict with keys:
    cat_size             : list[int]       vocab size per categorical feature
    regression_label     : str
    classification_label : str
    train_feat           : np.ndarray  (N, num_cat_cols)  int-encoded  LongTensor
    train_reg            : np.ndarray  (N,)               float32      regression targets
    train_cls            : np.ndarray  (N,)               int {0,1}    classification targets
    val_feat / val_reg / val_cls    : same shapes, validation split
    test_feat / test_reg / test_cls : same shapes, test split

Usage
-----
    from drug_demo_price_reg_rdmit_pytorch import data_preparation
    from mmoe_train import main
    data = data_preparation()
    main(data)
"""

import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════
# 1.  MODEL COMPONENTS
# ══════════════════════════════════════════════

class ExpertDNN(nn.Module):
    """Single expert MLP."""
    def __init__(self, input_dim: int, hidden_units: tuple, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class GateDNN(nn.Module):
    """Gating network — outputs a softmax mixture weight over experts."""
    def __init__(self, input_dim: int, gate_hidden: tuple, num_experts: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in gate_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_experts, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).softmax(dim=-1)   # (B, num_experts)


class TowerDNN(nn.Module):
    """Task-specific tower on top of the gated mixture output."""
    def __init__(self, input_dim: int, hidden_units: tuple, output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MMoEMultiTask(nn.Module):
    """
    Multi-gate Mixture-of-Experts for two tasks:
      task 0 – regression      (linear output, shape (B,))
      task 1 – classification  (sigmoid output, shape (B,))

    Parameters
    ----------
    embedding_dims : list of (vocab_size: int, emb_dim: int)
        One entry per categorical input feature.
        vocab_size should equal cat_size[i] from data_preparation().
        emb_dim is a hyperparameter (e.g. 4).
    num_experts    : int
    expert_hidden  : tuple of int   hidden layer sizes for each expert DNN
    gate_hidden    : tuple of int   hidden layer sizes for each gate DNN
    tower_hidden   : tuple of int   hidden layer sizes for each tower DNN
    dropout        : float

    forward(cat_inputs) -> (reg_out, cls_out)
    -----------------------------------------
    cat_inputs : LongTensor  (B, num_features)
        Each column i contains integer codes in [0, vocab_size_i].
    reg_out    : FloatTensor (B,)   raw regression score
    cls_out    : FloatTensor (B,)   probability in (0, 1)
    """

    def __init__(
        self,
        embedding_dims,
        num_experts: int = 4,
        expert_hidden: tuple = (128, 64),
        gate_hidden: tuple = (64,),
        tower_hidden: tuple = (32,),
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── per-feature embeddings ──
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab + 1, dim) for vocab, dim in embedding_dims
        ])
        input_dim = sum(dim for _, dim in embedding_dims)
        self.input_dropout = nn.Dropout(dropout)

        # ── shared expert DNNs ──
        self.experts = nn.ModuleList([
            ExpertDNN(input_dim, expert_hidden, dropout=dropout)
            for _ in range(num_experts)
        ])
        expert_out_dim = self.experts[0].out_dim

        # ── one gate per task ──
        self.gates = nn.ModuleList([
            GateDNN(input_dim, gate_hidden, num_experts, dropout=dropout)
            for _ in range(2)
        ])

        # ── task-specific towers ──
        self.tower_reg = TowerDNN(expert_out_dim, tower_hidden, 1, dropout=dropout)
        self.tower_cls = TowerDNN(expert_out_dim, tower_hidden, 1, dropout=dropout)

    def forward(self, cat_inputs):
        # embed and concatenate
        embs = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs, dim=-1)          # (B, input_dim)
        x = self.input_dropout(x)

        # expert outputs → (B, num_experts, expert_out_dim)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        def mix(gate):
            g = gate(x).unsqueeze(1)                    # (B, 1, num_experts)
            return torch.bmm(g, expert_outs).squeeze(1) # (B, expert_out_dim)

        reg_out = self.tower_reg(mix(self.gates[0])).squeeze(-1)                 # (B,)
        cls_out = torch.sigmoid(self.tower_cls(mix(self.gates[1]))).squeeze(-1)  # (B,)

        return reg_out, cls_out


# ══════════════════════════════════════════════
# 2.  DATASET
# ══════════════════════════════════════════════

class HealthcareDataset(Dataset):
    """
    Wraps the numpy arrays produced by data_preparation() into a PyTorch Dataset.

    Parameters
    ----------
    features   : np.ndarray (N, num_cat_cols)  int-encoded categorical features
    reg_labels : np.ndarray (N,)               float32 regression targets
    cls_labels : np.ndarray (N,)               int {0,1} classification targets
    """

    def __init__(self, features: np.ndarray, reg_labels: np.ndarray, cls_labels: np.ndarray):
        self.X   = torch.tensor(features,                dtype=torch.long)
        self.y_r = torch.tensor(reg_labels.astype("float32"), dtype=torch.float32)
        self.y_c = torch.tensor(cls_labels.astype("float32"), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_r[idx], self.y_c[idx]


def make_loaders(data: dict, batch_size: int = 64):
    """
    Build train / val / test DataLoaders from the data dict.

    Parameters
    ----------
    data       : dict returned by data_preparation()
    batch_size : int

    Returns
    -------
    train_dl, val_dl, test_dl : DataLoader
    """
    train_ds = HealthcareDataset(data["train_feat"], data["train_reg"], data["train_cls"])
    val_ds   = HealthcareDataset(data["val_feat"],   data["val_reg"],   data["val_cls"])
    test_ds  = HealthcareDataset(data["test_feat"],  data["test_reg"],  data["test_cls"])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)
    test_dl  = DataLoader(test_ds,  batch_size=256)

    return train_dl, val_dl, test_dl


# ══════════════════════════════════════════════
# 3.  METRICS
# ══════════════════════════════════════════════

def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Run inference over *loader* and return a metrics dict."""
    model.eval()
    all_reg_pred, all_cls_pred = [], []
    all_reg_true, all_cls_true = [], []

    with torch.no_grad():
        for X, yr, yc in loader:
            X = X.to(DEVICE)
            r, c = model(X)
            all_reg_pred.append(r.cpu().numpy())
            all_cls_pred.append(c.cpu().numpy())
            all_reg_true.append(yr.numpy())
            all_cls_true.append(yc.numpy())

    reg_pred = np.concatenate(all_reg_pred)
    cls_pred = np.concatenate(all_cls_pred)
    reg_true = np.concatenate(all_reg_true)
    cls_true = np.concatenate(all_cls_true)

    mse = mean_squared_error(reg_true, reg_pred)
    mae = mean_absolute_error(reg_true, reg_pred)
    r2  = r2_score(reg_true, reg_pred)
    auc = roc_auc_score(cls_true, cls_pred)

    cls_bin  = (cls_pred >= 0.5).astype(int)
    prec = precision_score(cls_true, cls_bin, zero_division=0)
    rec  = recall_score(cls_true,    cls_bin, zero_division=0)
    f1   = f1_score(cls_true,        cls_bin, zero_division=0)

    return dict(mse=mse, mae=mae, r2=r2, auc=auc, precision=prec, recall=rec, f1=f1)


def print_metrics(split: str, m: dict):
    print(
        f"[{split}]  "
        f"MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  |  "
        f"AUC={m['auc']:.4f}  Prec={m['precision']:.4f}  "
        f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}"
    )


# ══════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10,
) -> nn.Module:
    """
    Train *model* with combined MSE + BCE loss and early stopping.

    Parameters
    ----------
    model        : MMoEMultiTask  (already on DEVICE)
    train_loader : DataLoader     yields (X, y_reg, y_cls)
    val_loader   : DataLoader     same signature
    epochs       : int
    lr           : float          Adam learning rate
    patience     : int            early-stopping patience (epochs)

    Returns
    -------
    model with best-val-loss weights restored
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss  = nn.MSELoss()
    bce_loss  = nn.BCELoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X, yr, yc in train_loader:
            X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
            optimizer.zero_grad()
            reg_out, cls_out = model(X)
            loss = mse_loss(reg_out, yr) + bce_loss(cls_out, yc)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # validation loss
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X, yr, yc in val_loader:
                X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
                r, c = model(X)
                val_loss_sum += (mse_loss(r, yr) + bce_loss(c, yc)).item() * len(X)
        avg_val_loss = val_loss_sum / len(val_loader.dataset)

        train_m = evaluate(model, train_loader)
        val_m   = evaluate(model, val_loader)

        print(f"\nEpoch {epoch}/{epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")
        print_metrics("Train", train_m)
        print_metrics("Val  ", val_m)

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════
# 5.  HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════

def hyperparameter_search(data: dict, max_trials: int = 2) -> tuple:
    """
    Manual grid search over key hyperparameters (replaces Keras Tuner).

    Parameters
    ----------
    data       : dict from data_preparation()
    max_trials : int  number of HP combinations to try

    Returns
    -------
    (best_model, best_hp)
        best_model : trained MMoEMultiTask
        best_hp    : dict of best hyperparameter values
    """
    cat_size = data["cat_size"]

    # search grid
    grid = {
        "emb_dim":       [4],
        "num_experts":   [4, 8],
        "expert_hidden": [(128, 64)],
        "gate_hidden":   [(64,)],
        "tower_hidden":  [(32,)],
        "dropout":       [0.3, 0.4],
        "lr":            [1e-4, 1e-3],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))[:max_trials]

    train_dl, val_dl, _ = make_loaders(data, batch_size=64)

    best_val_loss = float("inf")
    best_model    = None
    best_hp       = None
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()

    for trial, combo in enumerate(combos, 1):
        hp = dict(zip(keys, combo))
        print(f"\n{'='*60}\nTrial {trial}/{len(combos)}  hp={hp}\n{'='*60}")

        embedding_dims = [(sz, hp["emb_dim"]) for sz in cat_size]
        model = MMoEMultiTask(
            embedding_dims = embedding_dims,
            num_experts    = hp["num_experts"],
            expert_hidden  = hp["expert_hidden"],
            gate_hidden    = hp["gate_hidden"],
            tower_hidden   = hp["tower_hidden"],
            dropout        = hp["dropout"],
        ).to(DEVICE)

        model = train_model(model, train_dl, val_dl, epochs=20, lr=hp["lr"], patience=5)

        # compute final val loss for comparison
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, yr, yc in val_dl:
                X, yr, yc = X.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
                r, c = model(X)
                vl += (mse_fn(r, yr) + bce_fn(c, yc)).item() * len(X)
        vl /= len(val_dl.dataset)

        if vl < best_val_loss:
            best_val_loss = vl
            best_model    = model
            best_hp       = hp

    print(f"\nBest hp: {best_hp}  (val_loss={best_val_loss:.4f})")
    return best_model, best_hp


# ══════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════

def main(data: dict = None):
    """
    Run HP search then final training and evaluation.

    Parameters
    ----------
    data : dict from data_preparation().
           If None, imports and calls data_preparation() automatically.
    """
    if data is None:
        from drug_demo_price_reg_rdmit_pytorch import data_preparation
        data = data_preparation()

    # ── hyperparameter search ──
    best_model, best_hp = hyperparameter_search(data, max_trials=2)

    # ── rebuild and retrain best config for more epochs ──
    cat_size       = data["cat_size"]
    embedding_dims = [(sz, best_hp["emb_dim"]) for sz in cat_size]

    final_model = MMoEMultiTask(
        embedding_dims = embedding_dims,
        num_experts    = best_hp["num_experts"],
        expert_hidden  = best_hp["expert_hidden"],
        gate_hidden    = best_hp["gate_hidden"],
        tower_hidden   = best_hp["tower_hidden"],
        dropout        = best_hp["dropout"],
    ).to(DEVICE)

    train_dl, val_dl, test_dl = make_loaders(data, batch_size=64)

    final_model = train_model(
        final_model, train_dl, val_dl,
        epochs=50, lr=best_hp["lr"], patience=10,
    )

    # ── final evaluation ──
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    print_metrics("Train", evaluate(final_model, train_dl))
    print_metrics("Val  ", evaluate(final_model, val_dl))
    print_metrics("Test ", evaluate(final_model, test_dl))


if __name__ == "__main__":
    main()
