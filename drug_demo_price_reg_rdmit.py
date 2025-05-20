"""
Multi-gate Mixture-of-Experts demo with healthcare data.
Modified to support both regression and classification tasks.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
Modified by [Your Name]
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
from tensorflow.keras.metrics import AUC, MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
from keras_tuner import Hyperband
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mmoe import MMoE

from keras_tuner import RandomSearch

SEED = 2

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.random.set_seed(SEED)

# Callback to print out evaluation metrics for both regression and classification tasks
class CustomCallback(Callback):
    def __init__(self, training_data, validation_data, test_data, regression_task, classification_task):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]
        self.regression_task = regression_task  # Name of regression task
        self.classification_task = classification_task  # Name of classification task

    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs['loss'] 
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        print(f"EPOCH {epoch+1}")
        for task_name in logs:
            if 'loss' in task_name:
                print(f"NAME: {task_name} -- LOSS: {logs[task_name]:.4f}")

        # Iterate through each task and output appropriate metrics
        for index, output_name in enumerate(self.model.output_names):
            if output_name == self.regression_task:
                # Regression metrics
                train_mse = mean_squared_error(self.train_Y[index], train_prediction[index])
                train_mae = mean_absolute_error(self.train_Y[index], train_prediction[index])
                train_r2 = r2_score(self.train_Y[index], train_prediction[index])
                
                validation_mse = mean_squared_error(self.validation_Y[index], validation_prediction[index])
                validation_mae = mean_absolute_error(self.validation_Y[index], validation_prediction[index])
                validation_r2 = r2_score(self.validation_Y[index], validation_prediction[index])
                
                test_mse = mean_squared_error(self.test_Y[index], test_prediction[index])
                test_mae = mean_absolute_error(self.test_Y[index], test_prediction[index])
                test_r2 = r2_score(self.test_Y[index], test_prediction[index])
                
                print(
                   'MSE-{}-Train: {:.4f} MAE-{}-Train: {:.4f} R²-{}-Train: {:.4f} \n'
                   'MSE-{}-Validation: {:.4f} MAE-{}-Validation: {:.4f} R²-{}-Validation: {:.4f} \n'
                   'MSE-{}-Test: {:.4f} MAE-{}-Test: {:.4f} R²-{}-Test: {:.4f}'.format(
                    output_name, train_mse, output_name, train_mae, output_name, train_r2,
                    output_name, validation_mse, output_name, validation_mae, output_name, validation_r2,
                    output_name, test_mse, output_name, test_mae, output_name, test_r2
                   )
                )
                
                print(f"Regression Task {output_name} statistics:")
                print(f"Train predictions - Mean: {np.mean(train_prediction[index]):.4f}, Min: {np.min(train_prediction[index]):.4f}, Max: {np.max(train_prediction[index]):.4f}")
                print(f"Train actual values - Mean: {np.mean(self.train_Y[index]):.4f}, Min: {np.min(self.train_Y[index]):.4f}, Max: {np.max(self.train_Y[index]):.4f}")
            
            else:
                # Classification metrics
                train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
                validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
                test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])

                threshold = 0.5
                y_pred_train = (train_prediction[index] >= threshold).astype(int)
                y_pred_validation = (validation_prediction[index] >= threshold).astype(int)
                y_pred_test = (test_prediction[index] >= threshold).astype(int)

                train_precision = precision_score(self.train_Y[index][:, 1], y_pred_train[:, 1])
                train_recall = recall_score(self.train_Y[index][:, 1], y_pred_train[:, 1])
                train_f1 = f1_score(self.train_Y[index][:, 1], y_pred_train[:, 1])

                validation_precision = precision_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1])
                validation_recall = recall_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1])
                validation_f1 = f1_score(self.validation_Y[index][:, 1], y_pred_validation[:, 1])

                test_precision = precision_score(self.test_Y[index][:, 1], y_pred_test[:, 1])
                test_recall = recall_score(self.test_Y[index][:, 1], y_pred_test[:, 1])
                test_f1 = f1_score(self.test_Y[index][:, 1], y_pred_test[:, 1])

                print(
                   'ROC-AUC-{}-Train: {:.4f} Precision-{}-Train: {:.4f} Recall-{}-Train: {:.4f} \n'
                   'ROC-AUC-{}-Validation: {:.4f} \n'
                   'ROC-AUC-{}-Test: {:.4f} Precision-{}-Test: {:.4f} Recall-{}-Test: {:.4f}'.format(
                    output_name, train_roc_auc,
                    output_name, train_precision,
                    output_name, train_recall,
                    output_name, validation_roc_auc,
                    output_name, test_roc_auc,
                    output_name, test_precision,
                    output_name, test_recall,
                   )
                )
                
                print(f"Classification Task {output_name} predictions:")
                print(f"Positive predictions count: {np.sum(y_pred_train[:, 1])}")
                print(f"Actual positives count: {np.sum(self.train_Y[index][:, 1])}")
                print(f"Prediction distribution: {np.mean(train_prediction[index][:, 1]):.4f}")
                print(f"Max prediction probability: {np.max(train_prediction[index][:, 1]):.4f}")

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
    regression_label = 'paid_amount'  # Changed from paid_more to paid_amount
    classification_label = 'RDMIT'

    label_columns = [regression_label, classification_label]
    
    categorical_columns = ['PROCTYP', 'CAP_SVC', 'FACPROF', 'MHSACOVG', 'NTWKPROV', 
                        'PAIDNTWK', 'ADMTYP', 'MDC', 'DSTATUS', 'PLANTYP', 'MSA', 'AGEGRP', 
                        'EECLASS', 'EESTATU', 'EMPREL', 'SEX', 'HLTHPLAN', 'INDSTRY','OUTPATIENT', 
                        'DEACLAS_x', 'GENIND_x', 'THERGRP_x', 'MAINTIN_y', 'PRODCAT', 
                        'SIGLSRC', 'GNINDDS', 'MAINTDS', 'PRDCTDS', 'EXCDGDS', 'MSTFMDS', 'THRCLDS', 
                        'THRGRDS', 'PHYFLAG', 'HOSP']

    # Following format of original code
    train_raw_labels = pd.read_csv("/content/keras-mmoe/data/train_raw_labels_pay_rdmit_3.csv.gz")
    other_raw_labels = pd.read_csv("/content/keras-mmoe/data/other_raw_labels_pay_rdmit_3.csv.gz") 
    transformed_train_main = pd.read_csv("/content/keras-mmoe/data/transformed_train_pay_rdmit_3.csv.gz") 
    transformed_other_main = pd.read_csv("/content/keras-mmoe/data/transformed_other_pay_rdmit_3.csv.gz") 

    transformed_train = transformed_train_main[categorical_columns]
    transformed_other = transformed_other_main[categorical_columns]

    cat_size_train = []
    transformed_train, cat_size = label_encode_df(transformed_train, cat_size_train)

    cat_size_other = []
    transformed_other, cat_size = label_encode_df(transformed_other, cat_size_other)

    cat_size = np.maximum(cat_size_train, cat_size_other)

    # payment amounts
    train_payment = train_raw_labels['PAY_PER_UNIT'].values  # Replace with actual payment amount column
    other_payment = other_raw_labels['PAY_PER_UNIT'].values  # Replace with actual payment amount column
    
    # Normalize payment
    scaler = StandardScaler()
    train_payment_scaled = scaler.fit_transform(train_payment.reshape(-1, 1)).flatten()
    other_payment_scaled = scaler.transform(other_payment.reshape(-1, 1)).flatten()
    
    train_RDMIT = to_categorical((train_raw_labels[classification_label] == 1).astype(int), num_classes=2)
    other_RDMIT = to_categorical((other_raw_labels[classification_label] == 1).astype(int), num_classes=2)

    dict_outputs = {
        regression_label: 1,  # 1 unit for regression output
        classification_label: train_RDMIT.shape[1]  # 2 units for classification output
    }
    
    dict_train_labels = {
        regression_label: train_payment_scaled,
        classification_label: train_RDMIT
    }
    
    dict_other_labels = {
        regression_label: other_payment_scaled,
        classification_label: other_RDMIT
    }
    
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    validation_indices = np.random.choice(len(other_payment_scaled), size=len(other_payment_scaled)//2, replace=False)
    test_indices = list(set(range(len(other_payment_scaled))) - set(validation_indices))
    
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [
        dict_other_labels[regression_label][validation_indices],  # Regression task 
        dict_other_labels[classification_label][validation_indices]  # Classification task
    ]
    
    test_data = transformed_other.iloc[test_indices]
    test_label = [
        dict_other_labels[regression_label][test_indices],  # Regression task
        dict_other_labels[classification_label][test_indices]  # Classification task
    ]
    
    train_data = transformed_train
    train_label = [
        dict_train_labels[regression_label],  # Regression task
        dict_train_labels[classification_label]  # Classification task
    ]

    return cat_size, regression_label, classification_label, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info, categorical_columns

def main():
    # Load the data
    cat_size, regression_label, classification_label, train_data, train_label, validation_data, validation_label, test_data, test_label, output_info, cat_cols = data_preparation()
    
    # Define the hyperparameter tuning process
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
      
        concat_layer = Concatenate()(embeddings)
        
        input_dropout_rate = hp.Float('input_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        concat_layer = Dropout(input_dropout_rate)(concat_layer)
        
        # MMoE layer
        mmoe_layers = MMoE(
            units=hp.Int('mmoe_units', min_value=4, max_value=8, step=2),
            num_experts=hp.Int('num_experts', min_value=4, max_value=8, step=2),
            num_tasks=2
        )(concat_layer)
        
        output_layers = []
        for index, (output_size, output_name) in enumerate(output_info):
            # Add dropout after MMoE layer for each task
            task_dropout_rate = hp.Float(f'task_{index}_dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
            task_layer = Dropout(task_dropout_rate)(mmoe_layers[index])
            
            tower_units = hp.Int(f'tower_units_task_{index}', min_value=2, max_value=8, step=2)
            tower_layer = Dense(
                units=tower_units,
                activation='relu',
                kernel_initializer=VarianceScaling(),
                kernel_regularizer=l2(0.01))(task_layer)
            
            # Add dropout after tower layer for each task
            tower_dropout_rate = hp.Float(f'tower_{index}_dropout_rate', min_value=0.3, max_value=0.5, step=0.1)
            tower_layer = Dropout(tower_dropout_rate)(tower_layer)
            
            # Different output activations and units based on task type
            if output_name == regression_label:
                # Linear activation for regression
                output_layer = Dense(
                    units=1,  # Single unit for regression
                    name=output_name,
                    activation='linear',  # Linear activation for regression
                    kernel_initializer=VarianceScaling(),
                    kernel_regularizer=l2(0.01))(tower_layer)
            else:
                # Softmax activation for classification
                output_layer = Dense(
                    units=output_size,
                    name=output_name,
                    activation='softmax',
                    kernel_initializer=VarianceScaling(),
                    kernel_regularizer=l2(0.01))(tower_layer)
                    
            output_layers.append(output_layer)
        
        # Different loss functions and metrics based on task type
        losses = {
            regression_label: 'mse',  # Mean squared error for regression
            classification_label: 'binary_crossentropy'  # Binary cross-entropy for classification
        }
        
        metrics = {
            regression_label: [MeanSquaredError(), MeanAbsoluteError(), RootMeanSquaredError()],
            classification_label: ['precision', AUC(name='roc_auc'), AUC(name='pr_auc', curve='PR')]
        }
        
        learning_rate = hp.Choice('learning_rate', values=[0.00001, 0.0001, 0.001])
        model = Model(inputs=inputs, outputs=output_layers)
        model.compile(
            loss=losses,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=metrics
        )

        return model

    train_inputs = [train_data.iloc[:, i].values for i in range(train_data.shape[1])]
    validation_inputs = [validation_data.iloc[:, i].values for i in range(validation_data.shape[1])]
    test_inputs = [test_data.iloc[:, i].values for i in range(test_data.shape[1])]
    
    # Used https://keras.io/keras_tuner/api/tuners/random/ 
    tuner = RandomSearch(
        build_model, 
        objective=[
            keras_tuner.Objective(f'val_{regression_label}_loss', direction='min'), 
            keras_tuner.Objective(f'val_{classification_label}_loss', direction='min')
        ],
        max_trials=2,
        directory='my_dir',
        project_name='mmoe_regression_classification_tuning', 
        overwrite=True
    )
    
    # Run the hyperparameter search 
    tuner.search(
        x=train_inputs,
        y={
            regression_label: train_label[0],  
            classification_label: train_label[1]
        },
        validation_data=(
            validation_inputs, 
            {
                regression_label: validation_label[0],
                classification_label: validation_label[1]
            }
        ),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=5,
                mode='min'
            )
        ],
        batch_size=64,
        epochs=20
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
        y={
            regression_label: train_label[0],
            classification_label: train_label[1]
        },
        validation_data=(
            validation_inputs, 
            {
                regression_label: validation_label[0],
                classification_label: validation_label[1]
            }
        ),
        epochs=50,
        callbacks=[
            CustomCallback(
                training_data=(train_inputs, [train_label[0], train_label[1]]),
                validation_data=(validation_inputs, [validation_label[0], validation_label[1]]),
                test_data=(test_inputs, [test_label[0], test_label[1]]),
                regression_task=regression_label,
                classification_task=classification_label
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=10,
                restore_best_weights=True,
                mode='min'
            )
        ],
        batch_size=64,
        shuffle=True
    )

    print("Final evaluation metrics:")
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    print(f"Final train loss: {train_loss[-1]:.4f}, val loss: {val_loss[-1]:.4f}")
    
    # Print regression metrics
    print(f"{regression_label} - Final train MSE: {history.history[f'{regression_label}_mean_squared_error'][-1]:.4f}")
    print(f"{regression_label} - Final val MSE: {history.history[f'val_{regression_label}_mean_squared_error'][-1]:.4f}")
    print(f"{regression_label} - Final train MAE: {history.history[f'{regression_label}_mean_absolute_error'][-1]:.4f}")
    print(f"{regression_label} - Final val MAE: {history.history[f'val_{regression_label}_mean_absolute_error'][-1]:.4f}")
    
    # Print classification metrics
    print(f"{classification_label} - Final train ROC-AUC: {history.history[f'{classification_label}_roc_auc'][-1]:.4f}")
    print(f"{classification_label} - Final val ROC-AUC: {history.history[f'val_{classification_label}_roc_auc'][-1]:.4f}")
    print(f"{classification_label} - Final train PR-AUC: {history.history[f'{classification_label}_pr_auc'][-1]:.4f}")
    print(f"{classification_label} - Final val PR-AUC: {history.history[f'val_{classification_label}_pr_auc'][-1]:.4f}")

if __name__ == '__main__':
    main()