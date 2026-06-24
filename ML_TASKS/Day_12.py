# ==========================================
# TASK 2 : WEATHER TYPE CLASSIFICATION
# USING ARTIFICIAL NEURAL NETWORK (ANN)
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("ML_TASKS/weather_classification_data.csv")

print("===================================")
print("DATASET INFORMATION")
print("===================================")

print("Dataset Shape :", data.shape)

print("\nFirst 5 Records")
print(data.head())

# ==========================================
# Features and Target
# ==========================================

X = data.drop("Weather Type", axis=1)
y = data["Weather Type"]

# ==========================================
# Convert Categorical Features
# ==========================================

# Automatically converts:
# Cloud Cover
# Season
# Location

X = pd.get_dummies(X, drop_first=True)

print("\nFeature Columns After Encoding:")
print(X.columns)

# Convert True/False to 0/1
X = X.astype(float)

# ==========================================
# Encode Target Variable
# ==========================================

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

print("\nWeather Classes:")
print(target_encoder.classes_)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Build ANN Model
# ==========================================

model = Sequential()

# Hidden Layer 1
model.add(
    Dense(
        units=16,
        activation='relu',
        input_shape=(X_train.shape[1],)
    )
)

# Hidden Layer 2
model.add(
    Dense(
        units=8,
        activation='relu'
    )
)

# Output Layer
model.add(
    Dense(
        units=len(np.unique(y)),
        activation='softmax'
    )
)

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

print("\n===================================")
print("MODEL SUMMARY")
print("===================================")

model.summary()

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20
)

# ==========================================
# Prediction
# ==========================================

y_pred_prob = model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(cm)

# ==========================================
# Accuracy Score
# ==========================================

acc = accuracy_score(y_test, y_pred)

print("\n===================================")
print("ACCURACY SCORE")
print("===================================")

print(acc)

# ==========================================
# Classification Report
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(classification_report(y_test, y_pred))

# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print(accuracy)

print("\n===================================")
print("MODEL LOSS")
print("===================================")

print(loss)