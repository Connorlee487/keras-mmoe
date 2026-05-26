import sys

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN, PredictionLayer

SEED = 2
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

REGRESSION_SCALE = 5.0


# -----------------------------------------------------------
# 1. Noisy Gate — mild tau + top-K sparse
# -----------------------------------------------------------
class NoisyGate(nn.Module):
    def __init__(self, input_dim, num_shared, num_task_specific,
                 noise_tau=0.3, top_k=3):
        super(NoisyGate, self).__init__()
        self.noise_tau      = noise_tau
        self.top_k          = top_k
        self.num_experts    = num_shared + num_task_specific
        self.w_gate         = nn.Linear(input_dim, self.num_experts, bias=False)
        self.w_noise        = nn.Linear(input_dim, self.num_experts, bias=False)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def forward(self, x):
        logits = self.w_gate(x)
        if self.training:
            sigma  = self.noise_tau * F.softplus(self.w_noise(x))
            eps    = sigma * torch.randn_like(sigma)
            logits = logits + eps

        topk_vals, _ = torch.topk(logits, self.top_k, dim=-1)
        threshold    = topk_vals[:, -1].unsqueeze(-1)
        mask         = logits < threshold
        logits       = logits.masked_fill(mask, float('-inf'))
        return F.softmax(logits, dim=-1)


# -----------------------------------------------------------
# 2. NG-MMoE with shared + task-specific experts
# -----------------------------------------------------------
class NGMMoE(BaseModel):
    def __init__(
        self,
        dnn_feature_columns,
        num_shared_experts=6,
        num_task_specific_experts=2,   # dedicated experts per task
        expert_dnn_hidden_units=(128,),
        tower_dnn_hidden_units=(128, 64),
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        l2_reg_dnn=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation='relu',
        task_types=('regression', 'binary'),
        task_names=('rating_prediction', 'watch_prediction'),
        device='cpu',
        noise_tau=0.3,
        top_k=3,
        load_balance_weight=0.005,
        gpus=None
    ):
        super(NGMMoE, self).__init__(
            linear_feature_columns=[],
            dnn_feature_columns=dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            init_std=init_std,
            seed=seed,
            device=device,
            gpus=gpus
        )

        self.num_shared          = num_shared_experts
        self.num_task_specific   = num_task_specific_experts
        self.num_tasks           = len(task_names)
        self.task_names          = task_names
        self.task_types          = task_types
        self.input_dim           = self.compute_input_dim(dnn_feature_columns)
        self.load_balance_weight = load_balance_weight

        # Shared experts — used by all tasks
        self.shared_experts = nn.ModuleList([
            DNN(self.input_dim, expert_dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=False,
                init_std=init_std, device=device)
            for _ in range(num_shared_experts)
        ])

        # Task-specific experts — one set per task
        # regression task gets its own experts, binary task gets its own
        self.task_specific_experts = nn.ModuleList([
            nn.ModuleList([
                DNN(self.input_dim, expert_dnn_hidden_units,
                    activation=dnn_activation, l2_reg=l2_reg_dnn,
                    dropout_rate=dnn_dropout, use_bn=False,
                    init_std=init_std, device=device)
                for _ in range(num_task_specific_experts)
            ])
            for _ in range(self.num_tasks)
        ])

        # Per-task noisy gates — routes over shared + that task's specific experts
        self.noisy_gates = nn.ModuleList([
            NoisyGate(self.input_dim,
                      num_shared_experts,
                      num_task_specific_experts,
                      noise_tau, top_k=top_k)
            for _ in range(self.num_tasks)
        ])

        self.tower_dnns = nn.ModuleList([
            DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=False,
                init_std=init_std, device=device)
            for _ in range(self.num_tasks)
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
            for _ in range(self.num_tasks)
        ])

        self.out_layers = nn.ModuleList([
            PredictionLayer(task_type)
            for task_type in task_types
        ])

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        # Shared expert outputs — (B, num_shared, hidden)
        shared_outputs = torch.stack(
            [e(dnn_input) for e in self.shared_experts], dim=1
        )

        task_outputs  = []
        # self.aux_loss = torch.tensor(0.0).to(dnn_input.device)
        self.balance_loss   = torch.tensor(0.0).to(dnn_input.device) 

        for k in range(self.num_tasks):
            # Task-specific expert outputs — (B, num_task_specific, hidden)
            specific_outputs = torch.stack(
                [e(dnn_input) for e in self.task_specific_experts[k]], dim=1
            )

            # Concatenate shared + task-specific for this task
            # (B, num_shared + num_task_specific, hidden)
            all_experts_k = torch.cat([shared_outputs, specific_outputs], dim=1)

            # Noisy gate over all experts for task k
            gate_k        = self.noisy_gates[k](dnn_input)   # (B, num_shared + num_specific)
            mean_load     = gate_k.mean(dim=0)
            #self.aux_loss = self.aux_loss + mean_load.var()
            self.balance_loss = self.balance_loss + mean_load.var()   

            fk        = (gate_k.unsqueeze(-1) * all_experts_k).sum(dim=1)
            tower_out = self.tower_dnns[k](fk)
            logit     = self.output_layers[k](tower_out)
            output    = self.out_layers[k](logit)
            task_outputs.append(output)

        return torch.cat(task_outputs, dim=-1)

    def get_regularization_loss(self):
        reg_loss = super(NGMMoE, self).get_regularization_loss()
        if hasattr(self, 'balance_loss'):
            reg_loss = reg_loss + self.load_balance_weight * self.balance_loss 
        return reg_loss

    def gate_entropy(self, X):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
                X, self.dnn_feature_columns, self.embedding_dict
            )
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            entropies = []
            for k in range(self.num_tasks):
                specific_outputs = torch.stack(
                    [e(dnn_input) for e in self.task_specific_experts[k]], dim=1
                )
                shared_outputs = torch.stack(
                    [e(dnn_input) for e in self.shared_experts], dim=1
                )
                _ = torch.cat([shared_outputs, specific_outputs], dim=1)
                g = self.noisy_gates[k](dnn_input)
                H = -(g * (g + 1e-8).log()).sum(dim=-1).mean()
                entropies.append(H.item())
        if was_training:
            self.train()
        return entropies


# -----------------------------------------------------------
# 3. Data — identical to MMOE script
# -----------------------------------------------------------
DATA_DIR = "./data/ml-100k"

ratings_df = pd.read_csv(
    os.path.join(DATA_DIR, "u.data"),
    sep="\t",
    names=["user_id", "movie_id", "user_rating", "timestamp"],
)

item_cols = [
    "movie_id", "movie_title", "release_date", "video_release_date", "imdb_url",
    "unknown","Action","Adventure","Animation","Childrens","Comedy",
    "Crime","Documentary","Drama","Fantasy","FilmNoir","Horror",
    "Musical","Mystery","Romance","SciFi","Thriller","War","Western",
]
genre_cols = [
    "unknown","Action","Adventure","Animation","Childrens","Comedy",
    "Crime","Documentary","Drama","Fantasy","FilmNoir","Horror",
    "Musical","Mystery","Romance","SciFi","Thriller","War","Western",
]
movies_df = pd.read_csv(
    os.path.join(DATA_DIR, "u.item"),
    sep="|", names=item_cols, encoding="latin-1",
)
def first_genre(row):
    for g in genre_cols:
        if row[g] == 1:
            return g
    return "unknown"
movies_df["genre"] = movies_df.apply(first_genre, axis=1)

users_df = pd.read_csv(
    os.path.join(DATA_DIR, "u.user"),
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip_code"],
    encoding="latin-1",
)

all_movie_ids   = movies_df["movie_id"].unique()
movie_title_map = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))

print(f"[INFO] Ratings : {len(ratings_df):,}")
print(f"[INFO] Users   : {ratings_df['user_id'].nunique()}")
print(f"[INFO] Movies  : {len(all_movie_ids)}")

pos_rows = []
neg_rows = []
for user_id, group in ratings_df.groupby("user_id"):
    watched_movie_ids = set(group["movie_id"].values)
    for _, row in group.iterrows():
        pos_rows.append({
            "user_id"    : user_id,
            "movie_id"   : row["movie_id"],
            "user_rating": float(row["user_rating"]),
            "watched"    : 1,
        })
    not_watched = [m for m in all_movie_ids if m not in watched_movie_ids]
    for m in not_watched:
        neg_rows.append({
            "user_id"    : user_id,
            "movie_id"   : m,
            "user_rating": 0.0,
            "watched"    : 0,
        })

pos_df = pd.DataFrame(pos_rows)
neg_df = pd.DataFrame(neg_rows).sample(n=len(pos_df), random_state=SEED)

data = pd.concat([pos_df, neg_df], ignore_index=True)\
         .sample(frac=1, random_state=SEED).reset_index(drop=True)

data = data.merge(users_df[["user_id","age","gender","occupation"]], on="user_id",  how="left")
data = data.merge(movies_df[["movie_id","movie_title","genre"]],     on="movie_id", how="left")

user_mean  = ratings_df.groupby("user_id")["user_rating"].mean().rename("user_mean_rating")
movie_mean = ratings_df.groupby("movie_id")["user_rating"].mean().rename("movie_mean_rating")
data = data.merge(user_mean,  on="user_id",  how="left")
data = data.merge(movie_mean, on="movie_id", how="left")
data["user_mean_rating"]  = data["user_mean_rating"].fillna(3.0)
data["movie_mean_rating"] = data["movie_mean_rating"].fillna(3.0)
data["age_norm"]          = data["age"] / data["age"].max()
data["expected_rating"]   = (data["user_mean_rating"] + data["movie_mean_rating"]) / 2.0

print(f"[INFO] Final dataset : {len(data):,} rows | "
      f"watched rate: {data['watched'].mean():.2%}")

data["rating_norm"] = data.apply(
    lambda row: (row["user_rating"]        / 5.0) * REGRESSION_SCALE if row["watched"] == 1
                else (row["movie_mean_rating"] / 5.0) * REGRESSION_SCALE,
    axis=1
)
target = ["rating_norm", "watched"]

sparse_features = ["user_id", "movie_title", "gender", "occupation", "genre"]
dense_features  = ["age_norm", "user_mean_rating", "movie_mean_rating", "expected_rating"]

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat].astype(str))

fixlen_feature_columns = (
    [SparseFeat(feat,
                vocabulary_size=int(data[feat].max()) + 1,
                embedding_dim=64)
     for feat in sparse_features]
    + [DenseFeat(feat, 1) for feat in dense_features]
)

dnn_feature_columns    = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names          = get_feature_names(linear_feature_columns + dnn_feature_columns)

split_boundary = int(len(data) * 0.8)
train = data.iloc[:split_boundary]
test  = data.iloc[split_boundary:]

train_model_input = {name: train[name].values for name in feature_names}
test_model_input  = {name: test[name].values  for name in feature_names}

# -----------------------------------------------------------
# 4. Model
# -----------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    print("[INFO] CUDA available – using GPU.")
    device = "cuda:0"

model = NGMMoE(
    dnn_feature_columns,
    task_types=["regression", "binary"],
    l2_reg_embedding=1e-5,
    task_names=["rating_prediction", "watch_prediction"],
    num_shared_experts=6,          # 6 shared experts
    num_task_specific_experts=2,   # 2 dedicated per task
    expert_dnn_hidden_units=(128,),
    tower_dnn_hidden_units=(128, 64),
    noise_tau=0.3,
    top_k=3,
    load_balance_weight=0.005,
    dnn_dropout=0.1,
    device=device,
)

model.compile(
    "adam",
    loss=["mse", "binary_crossentropy"],
    metrics=["mse"],
)

# -----------------------------------------------------------
# 5. Train
# -----------------------------------------------------------
epochs = int(os.getenv("DEEPCTR_EXAMPLE_EPOCHS", "10"))
print(f"\n[INFO] Training NGMMoE for {epochs} epoch(s) ...\n")
model.fit(
    train_model_input,
    train[target].values,
    batch_size=256,
    epochs=epochs,
    verbose=1,
)

# -----------------------------------------------------------
# 6. Evaluate
# -----------------------------------------------------------
print("\n[INFO] Evaluating on test set ...")
pred_ans = model.predict(test_model_input, batch_size=256)

watched_mask = test["watched"].values == 1
rmse = np.sqrt(mean_squared_error(
    test.loc[watched_mask, "rating_norm"].values / REGRESSION_SCALE,
    pred_ans[watched_mask, 0]                    / REGRESSION_SCALE,
)) * 5.0

ll  = round(log_loss(test["watched"].values, pred_ans[:, 1]), 4)
auc = round(roc_auc_score(test["watched"].values, pred_ans[:, 1]), 4)

print("\n── Test Results ──────────────────────────────────────────────────────")
print(f"  rating_prediction     RMSE (1-5 scale) = {rmse:.4f}")
print(f"  watch_prediction      LogLoss = {ll:.4f}  |  AUC = {auc:.4f}")
print("─────────────────────────────────────────────────────────────────────")

# -----------------------------------------------------------
# 7. Gate Entropy
# -----------------------------------------------------------
print("\n--- Gate Entropy ---")
sample_input = {name: test_model_input[name][:512] for name in test_model_input}

last_key   = list(model.feature_index.keys())[-1]
input_size = model.feature_index[last_key][1]
X_input    = torch.zeros(512, input_size).to(device)
for name in model.feature_index.keys():
    if name in sample_input:
        start, end = model.feature_index[name]
        X_input[:, start:end] = torch.tensor(
            sample_input[name][:512].reshape(-1, end - start),
            dtype=torch.float32
        ).to(device)

entropies   = model.gate_entropy(X_input)
max_entropy = np.log(8)
for i, task_name in enumerate(["rating_prediction", "watch_prediction"]):
    utilization = entropies[i] / max_entropy * 100
    print(f"Gate-Entropy-{task_name}: {entropies[i]:.4f} / {max_entropy:.4f} max  "
          f"({utilization:.1f}% utilization)")