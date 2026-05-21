# -*- coding: utf-8 -*-
import random
import os

import sys
sys.path.insert(0, '/projects/hps-gcnv-uswest8/xipang/DeepCTR-Torch-master')

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import MMOE

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


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
    train_df['income_label'] = (train_df['income_50k'] == ' 50000+.').astype(int)
    train_df['marital_label'] = (train_df['marital_stat'] == ' Never married').astype(int)
    other_df['income_label'] = (other_df['income_50k'] == ' 50000+.').astype(int)
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

    # Fit encoders on combined data for consistency
    combined_df = pd.concat([train_df, other_df], axis=0, ignore_index=True)

    for feat in sparse_features:
        lbe = LabelEncoder()
        combined_df[feat] = lbe.fit_transform(combined_df[feat].astype(str))

    mms = MinMaxScaler(feature_range=(0, 1))
    combined_df[dense_features] = mms.fit_transform(combined_df[dense_features])

    train_size = len(train_df)
    train_df = combined_df.iloc[:train_size].reset_index(drop=True)
    other_df = combined_df.iloc[train_size:].reset_index(drop=True)

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
    test_df = other_df.drop(validation_df.index)

    train_model_input = {name: train_df[name].values for name in feature_names}
    validation_model_input = {name: validation_df[name].values for name in feature_names}
    test_model_input = {name: test_df[name].values for name in feature_names}

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

    # MMoE: 8 experts, expert_dim=4, 2 tasks — matches original Keras config

    model = MMOE(
        dnn_feature_columns,
        task_types=['binary', 'binary'],
        l2_reg_embedding=1e-5,
        task_names=target,
        num_experts=8,
        expert_dnn_hidden_units=(4,),
        tower_dnn_hidden_units=(8,),
        device=device
    )
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

    # ROC-AUC evaluation across all splits
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


if __name__ == '__main__':
    main()