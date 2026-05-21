# -*- coding: utf-8 -*-
"""
Drug price regression + RDMIT readmission classification demo.
Modelled after mmoe_ex.py.

Uses the DeepCTR-Torch .compile() / .fit() / .predict() API with MMOE_MED.

Tasks
-----
  regression     : paid_amount  (PAY_PER_UNIT, z-score scaled)
  classification : RDMIT        (binary)
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from mmoe_med import MMOE_MED

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
REGRESSION_LABEL     = "paid_amount"
CLASSIFICATION_LABEL = "RDMIT"
TARGET               = [REGRESSION_LABEL, CLASSIFICATION_LABEL]

CATEGORICAL_COLUMNS = [
    "PROCTYP", "CAP_SVC", "FACPROF", "MHSACOVG", "NTWKPROV",
    "PAIDNTWK", "ADMTYP", "MDC", "DSTATUS", "PLANTYP", "MSA", "AGEGRP",
    "EECLASS", "EESTATU", "EMPREL", "SEX", "HLTHPLAN", "INDSTRY", "OUTPATIENT",
    "DEACLAS_x", "GENIND_x", "THERGRP_x", "MAINTIN_y", "PRODCAT",
    "SIGLSRC", "GNINDDS", "MAINTDS", "PRDCTDS", "EXCDGDS", "MSTFMDS", "THRCLDS",
    "THRGRDS", "PHYFLAG", "HOSP",
]

if __name__ == "__main__":

    # ── 1. Load raw data ──
    train_raw_labels       = pd.read_csv("/content/keras-mmoe/data/train_raw_labels_pay_rdmit_3.csv.gz")
    other_raw_labels       = pd.read_csv("/content/keras-mmoe/data/other_raw_labels_pay_rdmit_3.csv.gz")
    transformed_train_main = pd.read_csv("/content/keras-mmoe/data/transformed_train_pay_rdmit_3.csv.gz")
    transformed_other_main = pd.read_csv("/content/keras-mmoe/data/transformed_other_pay_rdmit_3.csv.gz")

    transformed_train = transformed_train_main[CATEGORICAL_COLUMNS].copy()
    transformed_other = transformed_other_main[CATEGORICAL_COLUMNS].copy()

    # ── 2. Label-encode sparse features; record per-split vocab sizes ──
    cat_size_train, cat_size_other = [], []

    for col in CATEGORICAL_COLUMNS:
        lbe_train = LabelEncoder()
        transformed_train[col] = lbe_train.fit_transform(transformed_train[col].astype(str))
        cat_size_train.append(len(lbe_train.classes_))

        lbe_other = LabelEncoder()
        transformed_other[col] = lbe_other.fit_transform(transformed_other[col].astype(str))
        cat_size_other.append(len(lbe_other.classes_))

    # vocab_size per feature = max across both splits so embeddings are large enough
    cat_size = np.maximum(cat_size_train, cat_size_other).tolist()

    # ── 3. Regression label: PAY_PER_UNIT z-score scaled ──
    train_payment = train_raw_labels["PAY_PER_UNIT"].values
    other_payment = other_raw_labels["PAY_PER_UNIT"].values

    scaler = StandardScaler()
    train_payment_scaled = scaler.fit_transform(train_payment.reshape(-1, 1)).flatten()
    other_payment_scaled = scaler.transform(other_payment.reshape(-1, 1)).flatten()

    # ── 4. Classification label: RDMIT binary 0/1 ──
    train_rdmit = (train_raw_labels[CLASSIFICATION_LABEL] == 1).astype(int).values
    other_rdmit = (other_raw_labels[CLASSIFICATION_LABEL] == 1).astype(int).values

    # ── 5. Attach labels to feature DataFrames ──
    transformed_train[REGRESSION_LABEL]     = train_payment_scaled
    transformed_train[CLASSIFICATION_LABEL] = train_rdmit

    transformed_other[REGRESSION_LABEL]     = other_payment_scaled
    transformed_other[CLASSIFICATION_LABEL] = other_rdmit

    # ── 6. Val / test split (50/50 from 'other') ──
    n_other  = len(transformed_other)
    val_idx  = np.random.choice(n_other, size=n_other // 2, replace=False)
    test_idx = np.array(list(set(range(n_other)) - set(val_idx)))

    train_data      = transformed_train
    validation_data = transformed_other.iloc[val_idx].reset_index(drop=True)
    test_data       = transformed_other.iloc[test_idx].reset_index(drop=True)

    # ── 7. Build DeepCTR-Torch feature columns ──
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=cat_size[i] + 1, embedding_dim=4)
        for i, feat in enumerate(CATEGORICAL_COLUMNS)
    ]

    dnn_feature_columns = fixlen_feature_columns
    feature_names       = get_feature_names(dnn_feature_columns)

    # ── 8. Build model inputs (dict of feature-name → array) ──
    train_model_input      = {name: train_data[name].values      for name in feature_names}
    validation_model_input = {name: validation_data[name].values for name in feature_names}
    test_model_input       = {name: test_data[name].values       for name in feature_names}

    # stack regression + classification labels → (N, 2)
    train_labels      = np.column_stack([train_data[TARGET[0]].values,
                                         train_data[TARGET[1]].values])
    validation_labels = np.column_stack([validation_data[TARGET[0]].values,
                                         validation_data[TARGET[1]].values])
    test_labels       = np.column_stack([test_data[TARGET[0]].values,
                                         test_data[TARGET[1]].values])

    # ── 9. Device ──
    device   = "cpu"
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print("cuda ready...")
        device = "cuda:0"

    # ── 10. Define model ──
    model = MMOE_MED(
        dnn_feature_columns,
        num_experts=4,
        expert_dnn_hidden_units=(128, 64),
        gate_dnn_hidden_units=(64,),
        tower_dnn_hidden_units=(32,),
        l2_reg_embedding=1e-5,
        task_types=["regression", "binary"],
        task_names=TARGET,
        device=device,
    )

    model.compile(
        "adam",
        loss=["mse", "binary_crossentropy"],
        metrics=["mse", "binary_crossentropy"],
    )

    # ── 11. Train ──
    epochs = int(os.getenv("DEEPCTR_EXAMPLE_EPOCHS", "10"))
    history = model.fit(
        train_model_input,
        train_labels,
        batch_size=64,
        epochs=epochs,
        verbose=2,
        validation_data=(validation_model_input, validation_labels),
    )

    # ── 12. Predict & evaluate ──
    pred_ans = model.predict(test_model_input, batch_size=256)   # (N_test, 2)

    print("")
    # Regression task
    reg_true = test_labels[:, 0]
    reg_pred = pred_ans[:, 0]
    print("{} test MSE".format(TARGET[0]), round(mean_squared_error(reg_true, reg_pred), 4))
    print("{} test MAE".format(TARGET[0]), round(mean_absolute_error(reg_true, reg_pred), 4))
    print("{} test R²".format(TARGET[0]),  round(r2_score(reg_true, reg_pred), 4))

    # Classification task
    cls_true = test_labels[:, 1]
    cls_pred = pred_ans[:, 1]
    print("{} test AUC".format(TARGET[1]), round(roc_auc_score(cls_true, cls_pred), 4))