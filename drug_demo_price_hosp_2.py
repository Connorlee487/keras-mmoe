"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import keras_tuner
import matplotlib.pyplot as plt

from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import AUC
from keras_tuner import Hyperband
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from mmoe import MMoE

from keras_tuner import RandomSearch

SEED = 2

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.random.set_seed(SEED)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):

        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs['loss'] 
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        print(f"EPOCH {epoch+1}")
        for task_name in logs:
            if 'loss' in task_name:
                print(f"NAME: {task_name} -- LOSS: {logs[task_name]:.4f}")

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])

            threshold = 0.5
            y_pred_train = (train_prediction[index] >= threshold).astype(int)
            y_pred_validation = (validation_prediction[index] >= threshold).astype(int)
            y_pred_test = (test_prediction[index] >= threshold).astype(int)

            train_precision = precision_score(self.train_Y[index][:, 1], y_pred_train[:, 1]) #, average="weighted")
            train_recall = recall_score(self.train_Y[index][:, 1], y_pred_train[:, 1]) #, average="weighted")
            train_f1 = f1_score(self.train_Y[index][:, 1], y_pred_train[:, 1]) #, average="weighted")    

            validation_precision = precision_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1]) #, average='weighted')
            validation_recall = recall_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1]) #, average='weighted')
            validation_f1 = f1_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1]) #, average='weighted')


            test_precision = precision_score(self.test_Y[index][:, 1], y_pred_test[:, 1]) #, average='weighted')
            test_recall = recall_score(self.test_Y[index][:, 1], y_pred_test[:, 1]) #, average='weighted')
            test_f1 = f1_score(self.test_Y[index][:, 1], y_pred_test[:, 1]) #, average='weighted')

            print(
               'ROC-AUC-{}-Train: {} Precision-{}-Train: {} Recall-{}-Train: {} \nROC-AUC-{}-Validation: {} \nROC-AUC-{}-Test: {} Precision-{}-Test: {} Recall-{}-Test: {} \n'.format(
                output_name, round(train_roc_auc, 4),
                output_name, round(train_precision, 4),
                output_name, round(train_recall, 4),

                output_name, round(validation_roc_auc, 4),

                output_name, round(test_roc_auc, 4),
                output_name, round(test_precision, 4),
                output_name, round(test_recall, 4),
               )
            )
            print(f"Task {output_name} predictions:")
            print(f"Positive predictions count: {np.sum(y_pred_train[:, 1])}")
            print(f"Actual positives count: {np.sum(self.train_Y[index][:, 1])}")
            print(f"Prediction distribution: {np.mean(train_prediction[index][:, 1]):.4f}")
            print(f"Max prediction probability: {np.max(train_prediction[index][:, 1]):.4f}")

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
def label_encode_df(df, sizes):
    label_encoders = {}

    for column in df.columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder
        sizes.append(len(encoder.classes_))
    return df, sizes

def data_preparation():
    label1 = 'paid_more'
    label2 = 'HOSP'

    label_columns = [label1, label2] #HOSP
    
    categorical_columns = ['PROCTYP', 'CAP_SVC', 'FACPROF', 'MHSACOVG', 'NTWKPROV', 
                        'PAIDNTWK', 'ADMTYP', 'MDC', 'DSTATUS', 'PLANTYP', 'MSA', 'AGEGRP', 
                        'EECLASS', 'EESTATU', 'EMPREL', 'SEX', 'HLTHPLAN', 'INDSTRY','OUTPATIENT', 
                        'DEACLAS_x', 'GENIND_x', 'THERGRP_x', 'MAINTIN_x', 'PRODCAT', 
                        'SIGLSRC', 'GNINDDS', 'MAINTDS', 'PRDCTDS', 'EXCDGDS', 'MSTFMDS', 'THRCLDS', 
                        'THRGRDS', 'PHYFLAG', 'RDMIT']
    
    # numerical_columns = ['PAY_PER_UNIT']
    # Following format of original code
    train_raw_labels = pd.read_csv("/content/keras-mmoe/data/train_raw_labels_pay_hosp_3.csv.gz")
    other_raw_labels = pd.read_csv("/content/keras-mmoe/data/other_raw_labels_pay_hosp_3.csv.gz") 
    transformed_train_main = pd.read_csv("/content/keras-mmoe/data/transformed_train_pay_hosp_3.csv.gz") 
    transformed_other_main = pd.read_csv("/content/keras-mmoe/data/transformed_other_pay_hosp_3.csv.gz") 

    transformed_train = transformed_train_main[categorical_columns]
    transformed_other = transformed_other_main[categorical_columns]

    # transformed_train_dense = transformed_train_main[numerical_columns]
    # transformed_other_dense = transformed_other_main[numerical_columns]


    cat_size_train = []
    transformed_train, cat_size = label_encode_df(transformed_train, cat_size_train)

    cat_size_other = []
    transformed_other, cat_size = label_encode_df(transformed_other, cat_size_other)

    # transformed_train = pd.concat([transformed_train, transformed_train_dense], axis=1)
    # transformed_other = pd.concat([transformed_other, transformed_other_dense], axis=1)


    cat_size = np.maximum(cat_size_train, cat_size_other)

    train_HOSP = to_categorical((train_raw_labels[label1] == 1).astype(int), num_classes=2)
    train_RDMIT = to_categorical((train_raw_labels[label2] == 1).astype(int), num_classes=2)

    other_HOSP = to_categorical((other_raw_labels[label1] == 1).astype(int), num_classes=2)
    other_RDMIT = to_categorical((other_raw_labels[label2] == 1).astype(int), num_classes=2)

    dict_outputs = {
        label1: train_HOSP.shape[1],
        label2: train_RDMIT.shape[1]
    }
    dict_train_labels = {
        label1: train_HOSP,
        label2: train_RDMIT
    }
    dict_other_labels = {
        label1: other_HOSP,
        label2: other_RDMIT
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    train_data = train_data[categorical_columns]# + numerical_columns]
    validation_data = validation_data[categorical_columns]# + numerical_columns]
    test_data = test_data[categorical_columns]# + numerical_columns]


    return cat_size, label1, label2, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info, categorical_columns #, numerical_columns

def main():
    # Load the data
    cat_size, label1, label2, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info, cat_cols = data_preparation()
    
    
    # Define the hyperparameter tuning process https://keras.io/keras_tuner/getting_started/ 
    def build_model(hp):
        embeddings = []
        inputs = []

        for i, size in enumerate(cat_size):
            input_layer = Input(shape=(1,), name=str(i))
            inputs.append(input_layer) 
        
            # Hyperparameter tuning
            embedding_dim = hp.Choice('embedding_dim', values=[4])
            embedding_layer = Embedding(input_dim=size + 1, output_dim=embedding_dim)(input_layer)
            embeddings.append(Flatten()(embedding_layer))

        # dense_input = Input(shape=(len(numerical_columns),), name="dense_input")

        # inputs.append(dense_input)
      
        concat_layer = Concatenate()(embeddings)
        
        # Add dropout layer after concatenation (before MMoE)
        input_dropout_rate = hp.Float('input_dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        concat_layer = Dropout(input_dropout_rate)(concat_layer)
        
        # MMoE layer
        mmoe_layers = MMoE(
            units=hp.Int('mmoe_units', min_value=2, max_value=8, step=2),
            num_experts=hp.Int('num_experts', min_value=2, max_value=8, step=2),
            num_tasks=2
        )(concat_layer)
        
        output_layers = []
        for index, task_layer in enumerate(mmoe_layers):
            # Add dropout after MMoE layer for each task
            task_dropout_rate = hp.Float(f'task_{index}_dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
            task_layer = Dropout(task_dropout_rate)(task_layer)
            
            tower_units = hp.Int(f'tower_units_task_{index}', min_value=2, max_value=8, step=2)
            tower_layer = Dense(
                units=tower_units,
                activation='relu',
                kernel_initializer=VarianceScaling(),
                kernel_regularizer=l2(0.01))(task_layer)
            
            # Add dropout after tower layer for each task
            tower_dropout_rate = hp.Float(f'tower_{index}_dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
            tower_layer = Dropout(tower_dropout_rate)(tower_layer)
            
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling(),
                kernel_regularizer=l2(0.01))(tower_layer)
            output_layers.append(output_layer)
        
        learning_rate = hp.Choice('learning_rate', values=[0.00001, 0.0001])
        model = Model(inputs=[inputs], outputs=output_layers)
        model.compile(
            loss={label1: 'binary_crossentropy', label2: 'binary_crossentropy'},
            optimizer=Adam(learning_rate=learning_rate),
            metrics={
              label1: ['precision', AUC(name='roc_auc'), AUC(name='pr_auc', curve='PR')],
              label2: ['precision', AUC(name='roc_auc'), AUC(name='pr_auc', curve='PR')]
            }
        )

        return model

    train_inputs = [train_data.iloc[:, i].values for i in range(train_data.shape[1])]
    validation_inputs = [validation_data.iloc[:, i].values for i in range(validation_data.shape[1])]
    test_inputs = [test_data.iloc[:, i].values for i in range(test_data.shape[1])]
    
    # Used https://keras.io/keras_tuner/api/tuners/random/ 
    tuner = RandomSearch(
        build_model, 
        objective=[keras_tuner.Objective('val_paid_more_loss', direction='min'), 
                  keras_tuner.Objective('val_RDMIT_loss', direction='min')], # Minimize loss for both
        max_trials=5,  # Increased the number of trials for better hyperparameter exploration
        directory='my_dir',
        project_name='mmoe_hyperparameter_tuning', 
        overwrite=True
    )
    
    # Run the hyperparameter search 
    tuner.search(
        x=train_inputs,
        y=train_label,
        validation_data=(validation_inputs, validation_label),
        # https://keras.io/api/callbacks/early_stopping/ 
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=5,
                mode='min'
            )
        ],
        batch_size=32,
        epochs=20  # Set a reasonable number of epochs for each trial
    )
    
    # Retrieve the best model and hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print("Best hyperparameters found:")
    for key, value in best_hps.values.items():
        print(f"{key}: {value}")
    
    # Train the best model with more epochs
    history = best_model.fit(
        x=train_inputs,
        y=train_label,
        validation_data=(validation_inputs, validation_label),
        epochs=50,
        callbacks=[
            ROCCallback(
                training_data=(train_inputs, train_label),
                validation_data=(validation_inputs, validation_label),
                test_data=(test_inputs, test_label)
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=10,
                restore_best_weights=True,
                mode='min'
            )
        ],
        batch_size=32,
    )

    print("Final evaluation metrics:")
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_pr_auc_HOSP = history.history['paid_more_pr_auc']
    val_pr_auc_HOSP = history.history['val_paid_more_pr_auc']

    train_roc_auc_HOSP = history.history['paid_more_roc_auc']
    val_roc_auc_HOSP = history.history['val_paid_more_roc_auc']

    train_pr_auc_RDMIT = history.history['RDMIT_pr_auc']
    val_pr_auc_RDMIT = history.history['val_RDMIT_pr_auc']

    train_roc_auc_RDMIT = history.history['RDMIT_roc_auc']
    val_roc_auc_RDMIT = history.history['val_RDMIT_roc_auc']

    
    print(f"Final train loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")
    print(f"paid_more - Final train ROC-AUC: {train_roc_auc_HOSP[-1]:.4f}, val ROC-AUC: {val_roc_auc_HOSP[-1]:.4f}")
    print(f"paid_more - Final train PR-AUC: {train_pr_auc_HOSP[-1]:.4f}, val PR-AUC: {val_pr_auc_HOSP[-1]:.4f}")
    print(f"RDMIT - Final train ROC-AUC: {train_roc_auc_RDMIT[-1]:.4f}, val ROC-AUC: {val_roc_auc_RDMIT[-1]:.4f}")
    print(f"RDMIT - Final train PR-AUC: {train_pr_auc_RDMIT[-1]:.4f}, val PR-AUC: {val_pr_auc_RDMIT[-1]:.4f}")


    print([train_loss, val_loss])
    print([train_pr_auc_HOSP,
    val_pr_auc_HOSP,
    train_roc_auc_HOSP,
    val_roc_auc_HOSP,
    train_pr_auc_RDMIT,
    val_pr_auc_RDMIT,
    train_roc_auc_RDMIT,
    val_roc_auc_RDMIT])


if __name__ == '__main__':
    main()