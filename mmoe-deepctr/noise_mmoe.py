# -*- coding: utf-8 -*-
import random
import os

import sys
sys.path.insert(0, '/projects/hps-gcnv-uswest8/xipang/DeepCTR-Torch-master')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import DNN, PredictionLayer

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------------------------------------
# 1. Noisy Gate (paper Eq. 3-4)
# -----------------------------------------------------------
class NoisyGate(nn.Module):
    """
    Training:
        logits = W_gate(x)
        sigma  = tau * softplus(W_noise(x))
        eps    = sigma * z,  z ~ N(0, I)   [reparameterization]
        output = softmax(logits + eps)
    Inference:
        output = softmax(W_gate(x))        [deterministic, zero overhead]
    """
    def __init__(self, input_dim, num_experts, noise_tau=1.0):
        super(NoisyGate, self).__init__()
        self.noise_tau = noise_tau
        self.w_gate  = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def forward(self, x):
        logits = self.w_gate(x)                                    # (B, n_experts)
        if self.training:
            sigma  = self.noise_tau * F.softplus(self.w_noise(x)) # (B, n_experts)
            eps    = sigma * torch.randn_like(sigma)
            logits = logits + eps
        return F.softmax(logits, dim=-1)                           # (B, n_experts)


# -----------------------------------------------------------
# 2. NG-MMoE built on DeepCTR-Torch BaseModel
#    Follows the same structure as DeepCTR-Torch MMOE
# -----------------------------------------------------------
class NGMMoE(BaseModel):
    """
    Noisy-Gate MMoE extending DeepCTR-Torch BaseModel.
    Replaces deterministic softmax gate with noisy gate (paper Eq. 3-4).

    Same __init__ signature as DeepCTR-Torch MMOE for drop-in replacement.
    """
    def __init__(
        self,
        dnn_feature_columns,
        num_experts=8,
        expert_dnn_hidden_units=(16,),
        tower_dnn_hidden_units=(8,),
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        l2_reg_dnn=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation='relu',
        task_types=('binary', 'binary'),
        task_names=('income_label', 'marital_label'),
        device='cpu',
        noise_tau=1.0,         # paper Sec 5.4 default tau=1.0
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

        self.num_experts    = num_experts
        self.num_tasks      = len(task_names)
        self.task_names     = task_names
        self.task_types     = task_types
        self.use_dnn        = len(dnn_feature_columns) > 0
        self.input_dim      = self.compute_input_dim(dnn_feature_columns)

        # Shared expert DNNs
        self.experts = nn.ModuleList([
            DNN(
                self.input_dim,
                expert_dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=False,
                init_std=init_std,
                device=device
            )
            for _ in range(num_experts)
        ])

        # Per-task noisy gates (replaces standard linear softmax gate)
        self.noisy_gates = nn.ModuleList([
            NoisyGate(self.input_dim, num_experts, noise_tau)
            for _ in range(self.num_tasks)
        ])

        # Per-task tower DNNs
        self.tower_dnns = nn.ModuleList([
            DNN(
                expert_dnn_hidden_units[-1],
                tower_dnn_hidden_units,
                activation=dnn_activation,
                l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout,
                use_bn=False,
                init_std=init_std,
                device=device
            )
            for _ in range(self.num_tasks)
        ])

        # Per-task output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
            for _ in range(self.num_tasks)
        ])

        # Per-task prediction layers (sigmoid for binary)
        self.out_layers = nn.ModuleList([
            PredictionLayer(task_type)
            for task_type in task_types
        ])

        self.to(device)

    def forward(self, X):
        # Build sparse + dense embedding input (same as DeepCTR-Torch)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # (B, input_dim)

        # All expert outputs -> (B, n_experts, expert_hidden_dim)
        expert_outputs = torch.stack(
            [expert(dnn_input) for expert in self.experts], dim=1
        )

        task_outputs = []
        for k in range(self.num_tasks):
            # Noisy gate weights for task k -> (B, n_experts)
            gate_k = self.noisy_gates[k](dnn_input).unsqueeze(-1)  # (B, n_experts, 1)

            # Gated mixture: f^k(x) = sum_i g~^k_i * f_i(x)
            fk = (gate_k * expert_outputs).sum(dim=1)              # (B, expert_hidden_dim)

            # Tower k
            tower_out = self.tower_dnns[k](fk)                    # (B, tower_hidden_dim)
            logit     = self.output_layers[k](tower_out)           # (B, 1)
            output    = self.out_layers[k](logit)                  # (B, 1) sigmoid
            task_outputs.append(output)

        task_outputs = torch.cat(task_outputs, dim=-1)             # (B, num_tasks)
        return task_outputs

    def gate_entropy(self, X):
        """
        Paper Sec 5.1 metric: H^k = -sum_i g^k_i * log(g^k_i)
        Higher entropy = more balanced expert utilization.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
                X, self.dnn_feature_columns, self.embedding_dict
            )
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            entropies = []
            for k in range(self.num_tasks):
                g = self.noisy_gates[k](dnn_input)                 # (B, n_experts)
                H = -(g * (g + 1e-8).log()).sum(dim=-1).mean()
                entropies.append(H.item())
        if was_training:
            self.train()
        return entropies


# -----------------------------------------------------------
# 3. Data Preparation  (identical to your working code)
# -----------------------------------------------------------
def data_preparation():
    column_names = [
        'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
        'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
        'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
        'det_hh_fam_stat', 'det_hh_summ', 'instance_weight', 'mig_chg_msa',
        'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt', 'num_emp',
        'fam_under_18', 'country_father', 'country_mother', 'country_self',
        'citizenship', 'own_or_self', 'vet_question', 'vet_benefits',
        'weeks_worked', 'year', 'income_50k'
    ]

    train_df = pd.read_csv(
        'data/census-income.data.gz',
        delimiter=',', header=None, index_col=None, names=column_names
    )
    other_df = pd.read_csv(
        'data/census-income.test.gz',
        delimiter=',', header=None, index_col=None, names=column_names
    )

    # Binary labels for two tasks
    train_df['income_label']  = (train_df['income_50k']  == ' 50000+.').astype(int)
    train_df['marital_label'] = (train_df['marital_stat'] == ' Never married').astype(int)
    other_df['income_label']  = (other_df['income_50k']  == ' 50000+.').astype(int)
    other_df['marital_label'] = (other_df['marital_stat'] == ' Never married').astype(int)

    target = ['income_label', 'marital_label']

    sparse_features = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'hs_college', 'major_ind_code', 'major_occ_code', 'race',
        'hisp_origin', 'sex', 'union_member', 'unemp_reason',
        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
        'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
        'mig_prev_sunbelt', 'fam_under_18', 'country_father',
        'country_mother', 'country_self', 'citizenship', 'vet_question',
        'marital_stat'
    ]
    dense_features = [
        'age', 'wage_per_hour', 'capital_gains', 'capital_losses',
        'stock_dividends', 'instance_weight', 'num_emp', 'weeks_worked', 'year'
    ]

    # Fit encoders on combined for consistency
    combined_df = pd.concat([train_df, other_df], axis=0, ignore_index=True)

    for feat in sparse_features:
        lbe = LabelEncoder()
        combined_df[feat] = lbe.fit_transform(combined_df[feat].astype(str))

    mms = MinMaxScaler(feature_range=(0, 1))
    combined_df[dense_features] = mms.fit_transform(combined_df[dense_features])

    train_size = len(train_df)
    train_df   = combined_df.iloc[:train_size].reset_index(drop=True)
    other_df   = combined_df.iloc[train_size:].reset_index(drop=True)

    fixlen_feature_columns = (
        [SparseFeat(feat, vocabulary_size=combined_df[feat].max() + 1, embedding_dim=4)
         for feat in sparse_features]
        + [DenseFeat(feat, 1) for feat in dense_features]
    )

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # Split other into 50% validation, 50% test
    validation_df = other_df.sample(frac=0.5, replace=False, random_state=SEED)
    test_df       = other_df.drop(validation_df.index)

    train_model_input      = {name: train_df[name].values      for name in feature_names}
    validation_model_input = {name: validation_df[name].values for name in feature_names}
    test_model_input       = {name: test_df[name].values       for name in feature_names}

    return (
        train_df, train_model_input,
        validation_df, validation_model_input,
        test_df, test_model_input,
        dnn_feature_columns, target
    )


# -----------------------------------------------------------
# 4. Main  (same structure as your working MMOE code)
# -----------------------------------------------------------
def main():
    (
        train_df, train_model_input,
        validation_df, validation_model_input,
        test_df, test_model_input,
        dnn_feature_columns, target
    ) = data_preparation()

    print('Training data shape   = {}'.format(train_df.shape))
    print('Validation data shape = {}'.format(validation_df.shape))
    print('Test data shape       = {}'.format(test_df.shape))

    device = 'cpu'
    if torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    # NG-MMoE: drop-in replacement for MMOE
    # only difference from your MMOE call: class=NGMMoE, extra noise_tau param
    model = NGMMoE(
        dnn_feature_columns,
        task_types=['binary', 'binary'],
        l2_reg_embedding=1e-5,
        task_names=target,
        num_experts=8,
        expert_dnn_hidden_units=(4,),
        tower_dnn_hidden_units=(8,),
        noise_tau=1.0,          # paper default tau=1.0
        device=device
    )

    # Identical compile/fit/predict API as DeepCTR-Torch MMOE
    model.compile(
        optimizer="adam",
        loss=["binary_crossentropy", "binary_crossentropy"],
        metrics=["binary_crossentropy"]
    )

    epochs = int(os.getenv("CENSUS_EPOCHS", "20"))
    model.fit(
        train_model_input,
        train_df[target].values,
        batch_size=256,
        epochs=epochs,
        verbose=2,
        validation_data=(validation_model_input, validation_df[target].values)
    )

    # ROC-AUC evaluation (identical to your original code)
    print("\n--- Evaluation ---")
    for split_name, model_input, df in [
        ("Train",      train_model_input,      train_df),
        ("Validation", validation_model_input, validation_df),
        ("Test",       test_model_input,        test_df),
    ]:
        pred = model.predict(model_input, batch_size=256)
        for i, task_name in enumerate(target):
            auc = roc_auc_score(df[task_name].values, pred[:, i])
            print("ROC-AUC-{}-{}: {}".format(task_name, split_name, round(auc, 4)))

# Gate entropy report (paper Sec 5.1)
    print("\n--- Gate Entropy (paper Sec 5.1) ---")
    sample_model_input = {
        name: test_model_input[name][:512] for name in test_model_input
    }

    model.eval()
    with torch.no_grad():
        # Use DeepCTR-Torch's own input conversion (same as inside model.predict)
        x_tensor = model.input_from_feature_columns
        sample_x = [
            torch.tensor(
                sample_model_input[name], dtype=torch.float32
            ).to(device)
            for name in model.feature_index.keys()
        ]
        # Stack into the format DeepCTR-Torch expects internally
        X_input = torch.zeros(
            512, model.feature_index[list(model.feature_index.keys())[-1]][1]
        ).to(device)
        for name in model.feature_index.keys():
            if name in sample_model_input:
                start, end = model.feature_index[name]
                X_input[:, start:end] = torch.tensor(
                    sample_model_input[name][:512].reshape(-1, end - start),
                    dtype=torch.float32
                ).to(device)

        entropies = model.gate_entropy(X_input)

    max_entropy = np.log(8)
    for i, name in enumerate(['income', 'marital']):
        utilization = entropies[i] / max_entropy * 100
        print("Gate-Entropy-{}: {:.4f} / {:.4f} max  ({:.1f}% utilization)".format(
            name, entropies[i], max_entropy, utilization
        ))

if __name__ == '__main__':
    main()
