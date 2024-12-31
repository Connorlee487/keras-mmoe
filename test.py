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