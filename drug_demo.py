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
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from keras_tuner import Hyperband
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from mmoe import MMoE

from keras_tuner import RandomSearch

SEED = 1

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
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])

            threshold = 0.5
            y_pred_train = (train_prediction[index] >= threshold).astype(int)
            y_pred_validation = (validation_prediction[index] >= threshold).astype(int)
            y_pred_test = (test_prediction[index] >= threshold).astype(int)

            train_precision = precision_score(self.train_Y[index], y_pred_train, average="weighted")
            train_recall = recall_score(self.train_Y[index], y_pred_train, average="weighted")
            train_f1 = f1_score(self.train_Y[index], y_pred_train, average="weighted")    

            validation_precision = precision_score(self.validation_Y[index], y_pred_validation, average='weighted')
            validation_recall = recall_score(self.validation_Y[index], y_pred_validation, average='weighted')
            validation_f1 = f1_score(self.validation_Y[index], y_pred_validation, average='weighted')


            test_precision = precision_score(self.test_Y[index], y_pred_test, average='weighted')
            test_recall = recall_score(self.test_Y[index], y_pred_test, average='weighted')
            test_f1 = f1_score(self.test_Y[index], y_pred_test, average='weighted')

            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {} // Precision-{}-Train: {} Recall-{}-Train: {} F1-{}-Train: {} // Precision-{}-Validation: {} Recall-{}-Validation: {} F1-{}-Validation: {} // Precision-{}-Test: {} Recall-{}-Test: {} F1-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4),

                    output_name, round(train_precision, 4),
                    output_name, round(train_recall, 4),
                    output_name, round(train_f1, 4),

                    output_name, round(validation_precision, 4),
                    output_name, round(validation_recall, 4),
                    output_name, round(validation_f1, 4),
                    
                    output_name, round(test_precision, 4),
                    output_name, round(test_recall, 4),
                    output_name, round(test_f1, 4)

                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def data_preparation():
    label1 = 'HOSP'
    label2 = 'RDMIT'

    label_columns = [label1, label2] #HOSP
    
    # categorical_columns = ['PROCTYP', 'YEAR', 'CAP_SVC', 'FACPROF', 'MHSACOVG', 'NTWKPROV',  'PAIDNTWK', 'ADMTYP', 'MDC', 'DSTATUS', 'PLANTYP', 'MSA', 'AGEGRP', 'EECLASS', 'EESTATU', 'EMPREL', 'SEX', 'HLTHPLAN', 'INDSTRY','OUTPATIENT', 'DEACLAS_x', 'GENIND_x', 'THERGRP_x', 'MAINTIN_y', 'PHYFLAG', 'PRODCAT', 'SIGLSRC', 'GNINDDS', 'MAINTDS', 'PRDCTDS', 'EXCDGDS', 'MSTFMDS', 'THRCLDS', 'THRGRDS', 'STDPROV', 'NETPAY_x']

    categorical_columns = ['PROCTYP', 'CAP_SVC', 'FACPROF', 'MHSACOVG', 'NTWKPROV', 
                        'PAIDNTWK', 'ADMTYP', 'MDC', 'DSTATUS', 'PLANTYP', 'MSA', 'AGEGRP', 
                        'EECLASS', 'EESTATU', 'EMPREL', 'SEX', 'HLTHPLAN', 'INDSTRY','OUTPATIENT', 
                        'DEACLAS_x', 'GENIND_x', 'THERGRP_x', 'MAINTIN_y', 'PRODCAT', 
                        'SIGLSRC', 'GNINDDS', 'MAINTDS', 'PRDCTDS', 'EXCDGDS', 'MSTFMDS', 'THRCLDS', 
                        'THRGRDS', 'STDPROV', 'PHYFLAG']

    train_raw_labels = pd.read_csv("/content/keras-mmoe/data/train_raw_labels.csv.gz")
    other_raw_labels = pd.read_csv("/content/keras-mmoe/data/other_raw_labels.csv.gz") 
    transformed_train = pd.read_csv("/content/keras-mmoe/data/transformed_train.csv.gz") 
    transformed_other = pd.read_csv("/content/keras-mmoe/data/transformed_other.csv.gz") 

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


    return label1, label2, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def main():
    # Load the data
    label1, label2, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    
    # Define the hyperparameter tuning process
    def build_model(hp):
        num_features = train_data.shape[1]
        input_layer = Input(shape=(num_features,))
        
        # Hyperparameter tuning
        embedding_dim = hp.Choice('embedding_dim', values=[8, 16, 32])
        embedding_layer = Embedding(input_dim=500, output_dim=embedding_dim)(input_layer)
        pooled_output = GlobalAveragePooling1D()(embedding_layer)
        
        # MMoE layer
        mmoe_layers = MMoE(
            units=hp.Int('mmoe_units', min_value=4, max_value=16, step=4),
            num_experts=hp.Int('num_experts', min_value=4, max_value=12, step=4),
            num_tasks=2
        )(pooled_output)
        
        output_layers = []
        for index, task_layer in enumerate(mmoe_layers):
            tower_units = hp.Int(f'tower_units_task_{index}', min_value=8, max_value=32, step=8)
            tower_layer = Dense(
                units=tower_units,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            
            output_layer = Dense(
                units=output_info[index][0],
                name=output_info[index][1],
                activation='softmax',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)
        
        # Compile the model
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model = Model(inputs=[input_layer], outputs=output_layers)
        model.compile(
            loss={label1: 'binary_crossentropy', label2: 'binary_crossentropy'},
            optimizer=Adam(learning_rate=learning_rate),
            metrics=[['auc', 'precision', 'recall'], ['auc', 'precision', 'recall']]
        )
        return model
    
    # Initialize the tuner
    tuner = RandomSearch(
        build_model,
        objective=[['auc', 'precision', 'recall'], ['auc', 'precision', 'recall']],
        max_trials=8,
        directory='my_dir',
        project_name='mmoe_hyperparameter_tuning'
    )
    
    # Run the hyperparameter search
    tuner.search(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],

    )
    
    # Retrieve the best model and hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    
    print("Best hyperparameters found:")
    for key, value in best_hps.values.items():
        print(f"{key}: {value}")
    
    # Train the best model further
    best_model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        epochs=100,
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ]
    )






if __name__ == '__main__':
    main()
