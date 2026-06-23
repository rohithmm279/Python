# ==========================================
# Artificial Neural Network (ANN)
# Churn Modelling Dataset
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

data = pd.read_csv("ML_TASKS/Churn_Modelling.csv")

print("Dataset Shape :", data.shape)
print(data.head())

# ==========================================
# Features and Target
# ==========================================

X = data.iloc[:, 3:-1]
y = data.iloc[:, -1]

# ==========================================
# Encoding Categorical Data
# ==========================================

# Geography -> One Hot Encoding
geo = pd.get_dummies(X["Geography"], drop_first=True)

# Gender -> Label Encoding
le = LabelEncoder()
X["Gender"] = le.fit_transform(X["Gender"])

# Remove Geography Column
X = X.drop("Geography", axis=1)

# Add Encoded Geography Columns
X = pd.concat([X, geo], axis=1)

print("\nFeature Columns:")
print(X.columns)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
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
model.add(Dense(
    units=12,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

# Hidden Layer 2
model.add(Dense(
    units=6,
    activation='relu'
))

# Output Layer
model.add(Dense(
    units=1,
    activation='sigmoid'
))

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

# ==========================================
# Train ANN
# ==========================================

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

# ==========================================
# Evaluation
# ==========================================

cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)

print("\n========================")
print("Confusion Matrix")
print("========================")
print(cm)

print("\n========================")
print("Accuracy Score")
print("========================")
print(acc)

print("\n========================")
print("Classification Report")
print("========================")
print(classification_report(y_test, y_pred))

# ==========================================
# Evaluate ANN
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n========================")
print("Model Accuracy")
print("========================")
print(accuracy)

print("\n========================")
print("Model Loss")
print("========================")
print(loss)