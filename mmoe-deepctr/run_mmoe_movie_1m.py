import sys

import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import MMOE

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

REGRESSION_SCALE = 5.0

# -----------------------------------------------------------
# 1. Data — MovieLens 1M
# -----------------------------------------------------------
DATA_DIR = "./ml-1m"

# ratings.dat : user_id :: movie_id :: rating :: timestamp
ratings_df = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"),
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "user_rating", "timestamp"],
)

# movies.dat : movie_id :: title :: genres (pipe-separated string)
movies_df = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["movie_id", "movie_title", "genres"],
    encoding="latin-1",
)
# 1M genres is a string like "Action|Comedy" — take first genre
movies_df["genre"] = movies_df["genres"].str.split("|").str[0]
movies_df = movies_df[["movie_id", "movie_title", "genre"]]

# users.dat : user_id :: gender :: age :: occupation :: zip
# NOTE: 1M column order is different from 100k (gender comes before age)
users_df = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"),
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip_code"],
    encoding="latin-1",
)

all_movie_ids   = movies_df["movie_id"].unique()
movie_title_map = dict(zip(movies_df["movie_id"], movies_df["movie_title"]))

print(f"[INFO] Ratings : {len(ratings_df):,}")
print(f"[INFO] Users   : {ratings_df['user_id'].nunique()}")
print(f"[INFO] Movies  : {len(all_movie_ids)}")

# ── watched=1 / watched=0 ─────────────────────────────────────────────────────
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

# ── Merge side features ───────────────────────────────────────────────────────
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

# ── Targets ───────────────────────────────────────────────────────────────────
data["rating_norm"] = data.apply(
    lambda row: (row["user_rating"]        / 5.0) * REGRESSION_SCALE if row["watched"] == 1
                else (row["movie_mean_rating"] / 5.0) * REGRESSION_SCALE,
    axis=1
)
target = ["rating_norm", "watched"]

# ── Encode features ───────────────────────────────────────────────────────────
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

# ── Train / test 80/20 ────────────────────────────────────────────────────────
split_boundary = int(len(data) * 0.8)
train = data.iloc[:split_boundary]
test  = data.iloc[split_boundary:]

train_model_input = {name: train[name].values for name in feature_names}
test_model_input  = {name: test[name].values  for name in feature_names}

# -----------------------------------------------------------
# 2. Model
# -----------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    print("[INFO] CUDA available – using GPU.")
    device = "cuda:0"

model = MMOE(
    dnn_feature_columns,
    task_types=["regression", "binary"],
    l2_reg_embedding=1e-5,
    task_names=["rating_prediction", "watch_prediction"],
    num_experts=8,
    expert_dnn_hidden_units=(128,),
    tower_dnn_hidden_units=(128, 64),
    device=device,
)

model.compile(
    "adam",
    loss=["mse", "binary_crossentropy"],
    metrics=["mse"],
)

# -----------------------------------------------------------
# 3. Train
# -----------------------------------------------------------
epochs = int(os.getenv("DEEPCTR_EXAMPLE_EPOCHS", "10"))
print(f"\n[INFO] Training MMOE for {epochs} epoch(s) ...\n")
model.fit(
    train_model_input,
    train[target].values,
    batch_size=256,
    epochs=epochs,
    verbose=1,
)

# -----------------------------------------------------------
# 4. Evaluate
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

