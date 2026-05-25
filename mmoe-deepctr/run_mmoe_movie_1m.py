import sys
sys.path.insert(0, '/projects/hps-gcnv-uswest8/xipang/DeepCTR-Torch-master')

import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import MMOE

# ===========================================================
# Reproducibility
# ===========================================================

SEED = 1

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

REGRESSION_SCALE = 5.0

# ===========================================================
# 1. Load MovieLens 1M
# ===========================================================

DATA_DIR = "./data/ml-1m"

ratings_df = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"),
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "user_rating", "timestamp"],
    encoding="latin-1",
)

movies_df = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["movie_id", "movie_title", "genres"],
    encoding="latin-1",
)

users_df = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"),
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip_code"],
    encoding="latin-1",
)

# 2. Feature Engineering

def first_genre(genres):
    if pd.isna(genres):
        return "unknown"

    genres = str(genres)

    if "|" in genres:
        return genres.split("|")[0]

    return genres

movies_df["genre"] = movies_df["genres"].apply(first_genre)

all_movie_ids = movies_df["movie_id"].unique()

print(f"[INFO] Ratings : {len(ratings_df):,}")
print(f"[INFO] Users   : {ratings_df['user_id'].nunique():,}")
print(f"[INFO] Movies  : {len(all_movie_ids):,}")

# 3. Negative Sampling
pos_df = ratings_df.copy()

pos_df["watched"] = 1

positive_set = set(
    zip(ratings_df["user_id"], ratings_df["movie_id"])
)

neg_rows = []

num_negatives = len(pos_df)

rng = np.random.default_rng(SEED)

user_ids = ratings_df["user_id"].unique()

attempts = 0

while len(neg_rows) < num_negatives and attempts < num_negatives * 10:

    u = rng.choice(user_ids)
    m = rng.choice(all_movie_ids)

    if (u, m) not in positive_set:

        neg_rows.append({
            "user_id": u,
            "movie_id": m,
            "user_rating": 0.0,
            "timestamp": 0,
            "watched": 0,
        })

        # Prevent duplicate negatives
        positive_set.add((u, m))

    attempts += 1

neg_df = pd.DataFrame(neg_rows)

data = pd.concat([pos_df, neg_df], ignore_index=True)

data = data.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# 4. Merge Side Features

data = data.merge(
    users_df[["user_id", "age", "gender", "occupation"]],
    on="user_id",
    how="left"
)

data = data.merge(
    movies_df[["movie_id", "genre"]],
    on="movie_id",
    how="left"
)

# 5. Statistical Features

user_mean = (
    ratings_df
    .groupby("user_id")["user_rating"]
    .mean()
    .rename("user_mean_rating")
)

movie_mean = (
    ratings_df
    .groupby("movie_id")["user_rating"]
    .mean()
    .rename("movie_mean_rating")
)

data = data.merge(
    user_mean,
    on="user_id",
    how="left"
)

data = data.merge(
    movie_mean,
    on="movie_id",
    how="left"
)

data["user_mean_rating"] = data["user_mean_rating"].fillna(3.0)

data["movie_mean_rating"] = data["movie_mean_rating"].fillna(3.0)

data["age_norm"] = data["age"] / data["age"].max()

data["expected_rating"] = (
    data["user_mean_rating"] +
    data["movie_mean_rating"]
) / 2.0

print(
    f"[INFO] Final dataset : {len(data):,} rows | "
    f"watched rate: {data['watched'].mean():.2%}"
)

# 6. Targets
# -----------------------------------------------------------
# Regression target
#
# watched=1:
#   use real rating
#
# watched=0:
#   use movie average rating
# -----------------------------------------------------------

data["rating_norm"] = data.apply(
    lambda row:
        (row["user_rating"] / 5.0) * REGRESSION_SCALE
        if row["watched"] == 1
        else (row["movie_mean_rating"] / 5.0) * REGRESSION_SCALE,
    axis=1
)

target = ["rating_norm", "watched"]

# 7. Encode Features

sparse_features = [
    "user_id",
    "movie_id",
    "gender",
    "occupation",
    "genre",
]

dense_features = [
    "age_norm",
    "user_mean_rating",
    "movie_mean_rating",
    "expected_rating",
]

for feat in sparse_features:

    lbe = LabelEncoder()

    data[feat] = lbe.fit_transform(
        data[feat].astype(str)
    )

# DeepCTR feature columns
fixlen_feature_columns = (

    [
        SparseFeat(
            feat,
            vocabulary_size=int(data[feat].max()) + 1,
            embedding_dim=64,
        )
        for feat in sparse_features
    ]

    +

    [
        DenseFeat(feat, 1)
        for feat in dense_features
    ]
)

linear_feature_columns = fixlen_feature_columns

dnn_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns
)

# 8. Train / Test Split

split_boundary = int(len(data) * 0.8)

train = data.iloc[:split_boundary]

test = data.iloc[split_boundary:]

train_model_input = {
    name: train[name].values
    for name in feature_names
}

test_model_input = {
    name: test[name].values
    for name in feature_names
}

# 9. MMOE Model

device = "cpu"

if torch.cuda.is_available():

    print("[INFO] CUDA available – using GPU.")

    device = "cuda:0"

model = MMOE(

    dnn_feature_columns,

    task_types=[
        "regression",
        "binary",
    ],

    task_names=[
        "rating_prediction",
        "watch_prediction",
    ],

    num_experts=8,

    expert_dnn_hidden_units=(128,),

    tower_dnn_hidden_units=(128, 64),

    l2_reg_embedding=1e-5,

    device=device,
)

# 10. Compile

model.compile(
    optimizer="adam",
    loss=[
        "mse",
        "binary_crossentropy",
    ],
    metrics=["mse"],
)

# 11. Train

epochs = int(
    os.getenv("DEEPCTR_EXAMPLE_EPOCHS", "10")
)

print(f"\n[INFO] Training MMOE for {epochs} epoch(s)...\n")

model.fit(
    train_model_input,
    train[target].values,
    batch_size=256,
    epochs=epochs,
    verbose=1,
)

# 12. Evaluate

print("\n[INFO] Evaluating on test set ...")

pred_ans = model.predict(
    test_model_input,
    batch_size=256
)

# Task 1: RMSE on watched movies only

watched_mask = test["watched"].values == 1

rmse = np.sqrt(

    mean_squared_error(

        test.loc[
            watched_mask,
            "rating_norm"
        ].values / REGRESSION_SCALE,

        pred_ans[
            watched_mask,
            0
        ] / REGRESSION_SCALE,
    )

) * 5.0

# Task 2: Binary classification metrics

ll = round(
    log_loss(
        test["watched"].values,
        pred_ans[:, 1]
    ),
    4
)

auc = round(
    roc_auc_score(
        test["watched"].values,
        pred_ans[:, 1]
    ),
    4
)

# 13. Results

print("\n── Test Results ─────────────────────────────────────────")

print(
    f"  rating_prediction     "
    f"RMSE (1-5 scale) = {rmse:.4f}"
)

print(
    f"  watch_prediction      "
    f"LogLoss = {ll:.4f}  |  AUC = {auc:.4f}"
)

print("─────────────────────────────────────────────────────────")