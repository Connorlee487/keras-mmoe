# -*- coding: utf-8 -*-
import random
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, mean_squared_error
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
    def __init__(self, input_dim, num_experts, noise_tau=1):
        super(NoisyGate, self).__init__()
        self.noise_tau = noise_tau
        self.w_gate  = nn.Linear(input_dim, num_experts, bias=False)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def forward(self, x):
        logits = self.w_gate(x)                                    # (B, n_experts)
        if self.training:
            # CHANGED TO VV sigma  = self.noise_tau * F.softplus(self.w_noise(x)) # (B, n_experts)
            sigma = torch.clamp(
                self.noise_tau * F.softplus(self.w_noise(x)),
                max=0.5
            )
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

# All 18 possible genres in MovieLens 1M
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]


def load_raw_data():
    ratings = pd.read_csv(
        'data/ml-1m/ratings.dat',
        sep='::', header=None, engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    users = pd.read_csv(
        'data/ml-1m/users.dat',
        sep='::', header=None, engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
    )
    movies = pd.read_csv(
        'data/ml-1m/movies.dat',
        sep='::', header=None, engine='python',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    return ratings, users, movies


def build_genre_features(movies):
    """One-hot encode genres; returns movies df with binary genre columns."""
    for genre in ALL_GENRES:
        movies[genre] = movies['genres'].apply(
            lambda g: 1 if genre in g.split('|') else 0
        ).astype(np.int8)
    return movies


def build_negative_samples(ratings, users, movies_with_genres, neg_ratio=1):
    """
    Popularity-based negative sampling: sample movies proportional to their
    rating frequency. Popular movies are more likely to be negatives, which is
    more realistic — a user consciously skipping a well-known movie is more
    informative than ignoring an obscure one.
    """
    user_ids     = ratings['user_id'].unique()
    positive_set = set(zip(ratings['user_id'], ratings['movie_id']))

    # Popularity weights — sample popular movies more often as negatives
    movie_popularity = ratings['movie_id'].value_counts(normalize=True)
    movie_ids_sorted = movie_popularity.sort_index().index.values
    movie_probs      = movie_popularity.sort_index().values  # sums to 1

    neg_rows  = []
    total_neg = int(len(ratings) * neg_ratio)
    rng       = np.random.default_rng(SEED)

    attempts = 0
    while len(neg_rows) < total_neg and attempts < total_neg * 20:
        u = rng.choice(user_ids)
        m = rng.choice(movie_ids_sorted, p=movie_probs)  # popularity-weighted
        if (u, m) not in positive_set:
            neg_rows.append({'user_id': u, 'movie_id': m,
                             'rating': 0.0, 'timestamp': 0})
            positive_set.add((u, m))
        attempts += 1

    neg_df = pd.DataFrame(neg_rows)
    ratings['watch_label'] = 1
    neg_df['watch_label']  = 0

    combined = pd.concat([ratings, neg_df], ignore_index=True)
    combined['rating_label'] = combined['rating'].astype(float)
    return combined


def data_preparation():
    ratings, users, movies = load_raw_data()
    movies = build_genre_features(movies)

    df = build_negative_samples(ratings, users, movies)

    # Merge user and movie features
    df = df.merge(users, on='user_id', how='left')
    df = df.merge(
        movies[['movie_id'] + ALL_GENRES],
        on='movie_id', how='left'
    )

    # ==================================================================
    # TASK COMBINATION OPTIONS — uncomment ONE block, comment the rest
    # ==================================================================

    # ------------------------------------------------------------------
    # OPTION A (CURRENT): watch prediction + rating regression
    #   Task 1 (binary):     did the user watch the movie?
    #   Task 2 (regression): what rating did they give? (scaled to 0-1)
    #   model compile:  loss=["binary_crossentropy", "mse"]
    #   task_types:     ['binary', 'regression']
    # ------------------------------------------------------------------
    df['rating_label'] = df['rating_label'] / 5.0
    target = ['watch_label', 'rating_label']

    # ------------------------------------------------------------------
    # OPTION B: rating regression + genre affinity classification
    #   Task 1 (regression): predict the user's rating (scaled 0-1)
    #   Task 2 (binary):     did the user watch a Drama movie?
    #                        swap 'Drama' for any genre in ALL_GENRES
    #   model compile:  loss=["mse", "binary_crossentropy"]
    #   task_types:     ['regression', 'binary']
    #
    # HOW IT WORKS:
    #   genre_label=1 if movie belongs to chosen genre AND was watched.
    #   Negatives (watch_label=0) get genre_label=0.
    #   rating_label scaled to [0,1]; negatives keep rating_label=0.
    # ------------------------------------------------------------------
    # GENRE_TASK = 'Drama'
    # df['rating_label'] = df['rating_label'] / 5.0
    # df['genre_label']  = df[GENRE_TASK].astype(np.float32)
    # df.loc[df['watch_label'] == 0, 'genre_label'] = 0.0
    # target = ['rating_label', 'genre_label']

    # ------------------------------------------------------------------
    # OPTION C: rating regression + like classification  (rating >= 4)
    #   Task 1 (regression): predict the user's rating (scaled 0-1)
    #   Task 2 (binary):     did the user LIKE the movie? (rating >= 4)
    #   model compile:  loss=["mse", "binary_crossentropy"]
    #   task_types:     ['regression', 'binary']
    
    # HOW IT WORKS:
    #   like_label=1 if rating>=4, else 0. Negatives get like_label=0.
    #   Both tasks share low-level taste/quality features; gates specialise
    #   on the full-scale regression vs. the liked/not-liked boundary.
    # ------------------------------------------------------------------
    # df['rating_label'] = df['rating_label'] / 5.0
    # df['like_label']   = (df['rating'] >= 4).astype(np.float32)
    # df.loc[df['watch_label'] == 0, 'like_label'] = 0.0
    # target = ['rating_label', 'like_label']
    # ==================================================================

    # ------------------------------------------------------------------
    # Feature definitions
    # ------------------------------------------------------------------
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features  = ['timestamp'] + ALL_GENRES  # genres as 0/1 dense cols

    # zip_code has too many values and mixed formats — encode as sparse
    df['zip_code'] = df['zip_code'].astype(str).str[:5]  # normalise length
    sparse_features.append('zip_code')

    # Label-encode sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat].astype(str))

    # Scale dense features
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    # ------------------------------------------------------------------
    # Train / validation / test split  (60 / 20 / 20)
    # ------------------------------------------------------------------
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    train_df      = df.iloc[:int(n * 0.6)].reset_index(drop=True)
    validation_df = df.iloc[int(n * 0.6):int(n * 0.8)].reset_index(drop=True)
    test_df       = df.iloc[int(n * 0.8):].reset_index(drop=True)

    # ------------------------------------------------------------------
    # DeepCTR feature columns
    # ------------------------------------------------------------------
    fixlen_feature_columns = (
        [SparseFeat(feat,
                    vocabulary_size=int(df[feat].max()) + 1,
                    embedding_dim=4)
         for feat in sparse_features]
        + [DenseFeat(feat, 1) for feat in dense_features]
    )

    dnn_feature_columns    = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

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


    # CHANGE FOR NMAYBE BETTER PERF
    # model = NGMMoE(
    #     dnn_feature_columns,
    #     task_types=['binary', 'regression'],
    #     l2_reg_embedding=1e-5,
    #     task_names=target,
    #     num_experts=8,
    #     expert_dnn_hidden_units=(4,),
    #     tower_dnn_hidden_units=(8,),
    #     noise_tau=1.0,          # paper default tau=1.0
    #     device=device
    # )

    model = NGMMoE(
        dnn_feature_columns,
        task_types=['binary', 'regression'],
        l2_reg_embedding=1e-5,
        task_names=target,
        num_experts=4,
        expert_dnn_hidden_units=(16, 8),
        tower_dnn_hidden_units=(8,),
        dnn_dropout=0.1,
        noise_tau=0.1,
        device=device
    )

    # Identical compile/fit/predict API as DeepCTR-Torch MMOE
    model.compile(
        optimizer="adam",
        loss=["binary_crossentropy", "mse"],
        metrics=[]
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

    print("\n--- Evaluation ---")
    for split_name, model_input, df in [
        ("Train",      train_model_input,      train_df),
        ("Validation", validation_model_input, validation_df),
        ("Test",       test_model_input,       test_df),
    ]:
        pred = model.predict(model_input, batch_size=256)

        # Task 1: watch_label — AUC (all samples, both pos and neg)
        auc = roc_auc_score(df['watch_label'].values, pred[:, 0])
        print(f"AUC-watch_label-{split_name}:  {round(auc, 4)}")

        # Task 2: rating_label — RMSE and MAE on POSITIVE (watched) samples only
        # (negatives have rating=0 which is not a real rating, exclude them)
        pos_mask = df['watch_label'].values == 1
        true_ratings = df['rating_label'].values[pos_mask] * 5.0   # scale back to 1-5
        pred_ratings = np.clip(pred[:, 1][pos_mask] * 5.0, 1.0, 5.0)  # clip to valid range

        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae  = np.mean(np.abs(true_ratings - pred_ratings))
        print(f"RMSE-rating_label-{split_name}: {round(rmse, 4)}")
        print(f"MAE-rating_label-{split_name}:  {round(mae, 4)}")
        print()

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
    for i, name in enumerate(['watch', 'rating']):
        utilization = entropies[i] / max_entropy * 100
        print("Gate-Entropy-{}: {:.4f} / {:.4f} max  ({:.1f}% utilization)".format(
            name, entropies[i], max_entropy, utilization
        ))

if __name__ == '__main__':
    main()