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
    # Also update task_types and loss in model.compile() to match.
    # ==================================================================

    # ------------------------------------------------------------------
    # OPTION A (CURRENT): watch prediction + rating regression
    #   Task 1 (binary):     did the user watch the movie?
    #   Task 2 (regression): what rating did they give? (scaled to 0-1)
    #   task_types: ['binary', 'regression']
    #   loss:       ['binary_crossentropy', 'mse']
    # ------------------------------------------------------------------
    df['rating_label'] = df['rating_label'] / 5.0
    target = ['watch_label', 'rating_label']

    # ------------------------------------------------------------------
    # OPTION B: rating regression + genre affinity classification
    #   Task 1 (regression): predict the user's rating (scaled 0-1)
    #   Task 2 (binary):     did the user watch a Drama movie?
    #                        swap 'Drama' for any genre in ALL_GENRES
    #   task_types: ['regression', 'binary']
    #   loss:       ['mse', 'binary_crossentropy']
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
    #   task_types: ['regression', 'binary']
    #   loss:       ['mse', 'binary_crossentropy']
    #
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
    # MMoE config — UPDATE task_types and loss together when switching
    # task option above:
    #   Option A: task_types=['binary','regression'], loss=['binary_crossentropy','mse']
    #   Option B: task_types=['regression','binary'], loss=['mse','binary_crossentropy']
    #   Option C: task_types=['regression','binary'], loss=['mse','binary_crossentropy']
    # ------------------------------------------------------------------
    TASK_TYPES = ['binary', 'regression']   # <-- change with task option
    TASK_LOSS  = ['binary_crossentropy', 'mse']  # <-- change with task option

    model = MMOE(
        dnn_feature_columns,
        task_types=TASK_TYPES,
        l2_reg_embedding=1e-5,
        task_names=target,
        num_experts=4,
        expert_dnn_hidden_units=(16, 8),
        tower_dnn_hidden_units=(8,),
        device=device
    )
    model.compile(
        optimizer="adam",
        loss=TASK_LOSS,
        metrics=[]  # see note: DeepCTR applies metrics to all tasks combined,
                    # which breaks for mixed binary/regression. Evaluate manually below.
    )

    epochs = int(os.getenv("MOVIELENS_EPOCHS", "20"))

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
        pos_mask = df['watch_label'].values == 1  # only real watched rows

        for i, (task_name, task_type) in enumerate(zip(target, TASK_TYPES)):
            y_true = df[task_name].values
            y_pred = pred[:, i]

            if task_type == 'binary':
                # AUC over all samples (positives + negatives)
                auc = roc_auc_score(y_true, y_pred)
                print(f"AUC-{task_name}-{split_name}: {round(auc, 4)}")

            elif task_type == 'regression':
                # RMSE and MAE on watched rows only (negatives have label=0, not real)
                true_r = y_true[pos_mask] * 5.0              # scale back to 1-5
                pred_r = np.clip(y_pred[pos_mask] * 5.0, 1.0, 5.0)
                rmse = np.sqrt(mean_squared_error(true_r, pred_r))
                mae  = np.mean(np.abs(true_r - pred_r))
                print(f"RMSE-{task_name}-{split_name}: {round(rmse, 4)}")
                print(f"MAE-{task_name}-{split_name}:  {round(mae, 4)}")
        print()

if __name__ == '__main__':
    main()