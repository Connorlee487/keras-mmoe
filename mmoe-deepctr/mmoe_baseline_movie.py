# -*- coding: utf-8 -*-
"""
MMoE experiment on MovieLens 1M dataset.

Two tasks:
  Task 1 (binary):     watch_label  — did the user watch the movie?
                       Since every row in ratings.dat IS a watch event,
                       we generate negative samples by pairing users with
                       movies they have NOT rated (label=0). Positive rows
                       keep label=1.
  Task 2 (regression): rating_label — the user's rating (1–5, float).
                       For negative (unobserved) samples the rating is set
                       to 0.0 (they are masked out of the regression loss
                       via sample_weight if your framework supports it;
                       otherwise just leave as-is and note the limitation).

Data files expected (MovieLens 1M format, '::' separated):
  data/ratings.dat   — UserID::MovieID::Rating::Timestamp
  data/users.dat     — UserID::Gender::Age::Occupation::Zip-code
  data/movies.dat    — MovieID::Title::Genres
"""

import random
import os
import sys

sys.path.insert(0, '/projects/hps-gcnv-uswest8/xipang/DeepCTR-Torch-master')

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import MMOE

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

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
    For each positive (user, movie) pair, sample `neg_ratio` unobserved
    (user, movie) pairs as negatives.  Negatives get watch_label=0, rating=0.
    """
    user_ids = ratings['user_id'].unique()
    movie_ids = ratings['movie_id'].unique()
    positive_set = set(zip(ratings['user_id'], ratings['movie_id']))

    neg_rows = []
    total_neg = int(len(ratings) * neg_ratio)
    rng = np.random.default_rng(SEED)

    attempts = 0
    while len(neg_rows) < total_neg and attempts < total_neg * 20:
        u = rng.choice(user_ids)
        m = rng.choice(movie_ids)
        if (u, m) not in positive_set:
            neg_rows.append({'user_id': u, 'movie_id': m,
                             'rating': 0.0, 'timestamp': 0})
            positive_set.add((u, m))
        attempts += 1

    neg_df = pd.DataFrame(neg_rows)
    ratings['watch_label'] = 1
    neg_df['watch_label'] = 0

    combined = pd.concat([ratings, neg_df], ignore_index=True)
    # rating_label is the float rating (0 for negatives)
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

    df['rating_label'] = df['rating_label'] / 5.0

    target = ['watch_label', 'rating_label']

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

    # ------------------------------------------------------------------
    # MMoE — task_types: first task binary, second task regression
    # ------------------------------------------------------------------
    model = MMOE(
        dnn_feature_columns,
        task_types=['binary', 'regression'],
        l2_reg_embedding=1e-5,
        task_names=target,
        num_experts=4,
        expert_dnn_hidden_units=(16, 8),
        tower_dnn_hidden_units=(8,),
        device=device
    )
    model.compile(
        optimizer="adam",
        loss=["binary_crossentropy", "mse"],
        metrics=[]
    )

    epochs = int(os.getenv("MOVIELENS_EPOCHS", "30"))

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

if __name__ == '__main__':
    main()