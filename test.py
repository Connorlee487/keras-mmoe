import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

# Example dataset
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Feature1': [1.0, 2.1, 3.2, 1.4, 2.2, 3.3],
    'Feature2': [5.2, 4.1, 3.3, 5.5, 4.4, 3.5],
    'Label': [0, 1, 0, 0, 1, 1]
})

# Split data into features and labels
X = data[['Category', 'Feature1', 'Feature2']]
y = data['Label']

# Encode categorical input (One-Hot Encoding)
ohe = OneHotEncoder(sparse=False)
X_category = ohe.fit_transform(X[['Category']])

# Combine with other features
X_features = np.hstack([X_category, X[['Feature1', 'Feature2']].values])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Model definition
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# FOR CENSUS INCOME SAMPLE DATA 
"""
Epoch 1/100
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 9s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step
ROC-AUC-income-Train: 0.5 ROC-AUC-income-Validation: 0.5 ROC-AUC-income-Test: 0.5
ROC-AUC-marital-Train: 0.9744 ROC-AUC-marital-Validation: 0.9601 ROC-AUC-marital-Test: 0.9605
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 64s 10ms/step - income_auc: 0.9222 - income_loss: 0.6487 - loss: 1.3682 - marital_auc: 0.7332 - marital_loss: 0.7195 - val_income_auc: 0.9378 - val_income_loss: 0.2330 - val_loss: 0.4902 - val_marital_auc: 0.9590 - val_marital_loss: 0.2571
Epoch 2/100
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 9s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
ROC-AUC-income-Train: 0.5031 ROC-AUC-income-Validation: 0.507 ROC-AUC-income-Test: 0.5069
ROC-AUC-marital-Train: 0.99 ROC-AUC-marital-Validation: 0.9752 ROC-AUC-marital-Test: 0.9754
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 40s 6ms/step - income_auc: 0.9376 - income_loss: 0.2336 - loss: 0.4176 - marital_auc: 0.9772 - marital_loss: 0.1840 - val_income_auc: 0.9378 - val_income_loss: 0.2330 - val_loss: 0.4554 - val_marital_auc: 0.9651 - val_marital_loss: 0.2223
Epoch 3/100
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 8s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step
ROC-AUC-income-Train: 0.5032 ROC-AUC-income-Validation: 0.5062 ROC-AUC-income-Test: 0.5062
ROC-AUC-marital-Train: 0.9906 ROC-AUC-marital-Validation: 0.9737 ROC-AUC-marital-Test: 0.9738
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 50s 8ms/step - income_auc: 0.9376 - income_loss: 0.2336 - loss: 0.3899 - marital_auc: 0.9839 - marital_loss: 0.1563 - val_income_auc: 0.9378 - val_income_loss: 0.2330 - val_loss: 0.4709 - val_marital_auc: 0.9610 - val_marital_loss: 0.2379
Epoch 4/100
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 9s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1559/1559 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
ROC-AUC-income-Train: 0.5 ROC-AUC-income-Validation: 0.5 ROC-AUC-income-Test: 0.5
ROC-AUC-marital-Train: 0.9862 ROC-AUC-marital-Validation: 0.9651 ROC-AUC-marital-Test: 0.9647
6236/6236 ━━━━━━━━━━━━━━━━━━━━ 50s 8ms/step - income_auc: 0.9376 - income_loss: 0.2337 - loss: 0.3763 - marital_auc: 0.9856 - marital_loss: 0.1426 - val_income_auc: 0.9378 - val_income_loss: 0.2330 - val_loss: 0.5352 - val_marital_auc: 0.9489 - val_marital_loss: 0.3021
Epoch 5/100
"""

# FOR MEDICAL DATA 
"""
Epoch 1/5
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 10s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step
ROC-AUC-PHYFLAG-Train: 0.5 ROC-AUC-PHYFLAG-Validation: 0.5002 ROC-AUC-PHYFLAG-Test: 0.5001
ROC-AUC-RDMIT-Train: 0.5 ROC-AUC-RDMIT-Validation: 0.5 ROC-AUC-RDMIT-Test: 0.4999
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 58s 7ms/step - PHYFLAG_auc: 0.8229 - PHYFLAG_loss: 5.8808 - RDMIT_auc: 0.5099 - RDMIT_loss: 6.3763 - loss: 12.2571 - val_PHYFLAG_auc: 0.8248 - val_PHYFLAG_loss: 0.4640 - val_RDMIT_auc: 0.5096 - val_RDMIT_loss: 0.6930 - val_loss: 1.1571
Epoch 2/5
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 10s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
ROC-AUC-PHYFLAG-Train: 0.5 ROC-AUC-PHYFLAG-Validation: 0.5005 ROC-AUC-PHYFLAG-Test: 0.5001
ROC-AUC-RDMIT-Train: 0.5 ROC-AUC-RDMIT-Validation: 0.5 ROC-AUC-RDMIT-Test: 0.4997
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 75s 7ms/step - PHYFLAG_auc: 0.8257 - PHYFLAG_loss: 0.7123 - RDMIT_auc: 0.5101 - RDMIT_loss: 0.6930 - loss: 1.4053 - val_PHYFLAG_auc: 0.8248 - val_PHYFLAG_loss: 0.4640 - val_RDMIT_auc: 0.5096 - val_RDMIT_loss: 0.6930 - val_loss: 1.1571
Epoch 3/5
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 10s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
ROC-AUC-PHYFLAG-Train: 0.5 ROC-AUC-PHYFLAG-Validation: 0.5009 ROC-AUC-PHYFLAG-Test: 0.5006
ROC-AUC-RDMIT-Train: 0.5001 ROC-AUC-RDMIT-Validation: 0.5001 ROC-AUC-RDMIT-Test: 0.4999
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 78s 6ms/step - PHYFLAG_auc: 0.8260 - PHYFLAG_loss: 0.5716 - RDMIT_auc: 0.5101 - RDMIT_loss: 0.6930 - loss: 1.2646 - val_PHYFLAG_auc: 0.8248 - val_PHYFLAG_loss: 0.4640 - val_RDMIT_auc: 0.5096 - val_RDMIT_loss: 0.6930 - val_loss: 1.1571
Epoch 4/5
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 9s 1ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step
1585/1585 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
ROC-AUC-PHYFLAG-Train: 0.5 ROC-AUC-PHYFLAG-Validation: 0.5007 ROC-AUC-PHYFLAG-Test: 0.5003
ROC-AUC-RDMIT-Train: 0.5001 ROC-AUC-RDMIT-Validation: 0.5 ROC-AUC-RDMIT-Test: 0.4999
7394/7394 ━━━━━━━━━━━━━━━━━━━━ 87s 7ms/step - PHYFLAG_auc: 0.8260 - PHYFLAG_loss: 0.4616 - RDMIT_auc: 0.5101 - RDMIT_loss: 0.6930 - loss: 1.1546 - val_PHYFLAG_auc: 0.8248 - val_PHYFLAG_loss: 0.4640 - val_RDMIT_auc: 0.5096 - val_RDMIT_loss: 0.6930 - val_loss: 1.1571
Epoch 5/5
"""