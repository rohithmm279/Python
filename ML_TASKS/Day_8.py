# ==========================================
# WEATHER TYPE CLASSIFICATION
# RANDOM FOREST
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(r'R:\S5 AI roadmap\weather_classification_data.csv')

print("Dataset Shape:", df.shape)

print(df.head())

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("Weather Type", axis=1)

y = df["Weather Type"]

# ==========================================
# ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)

# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

categorical_cols = X.select_dtypes(
    include=['object', 'string']
).columns

numerical_cols = X.select_dtypes(
    exclude=['object']
).columns

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# ==========================================
# PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        ),
        (
            'num',
            'passthrough',
            numerical_cols
        )
    ]
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ==========================================
# MODEL
# ==========================================

model = Pipeline([
    ('preprocessor', preprocessor),

    ('classifier',
     RandomForestClassifier(
         n_estimators=200,
         random_state=42
     ))
])

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

rf = model.named_steps['classifier']

feature_names = model.named_steps[
    'preprocessor'
].get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 15 Important Features")

print(
    importance_df.head(15)
)

# ==========================================
# FEATURE IMPORTANCE PLOT
# ==========================================

plt.figure(figsize=(10,6))

top_features = importance_df.head(15)

sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature'
)

plt.title(
    "Top 15 Important Features"
)

plt.show()

# ==========================================
# SAVE RESULTS
# ==========================================

predictions = pd.DataFrame({
    "Actual":
    label_encoder.inverse_transform(y_test),

    "Predicted":
    label_encoder.inverse_transform(y_pred)
})

predictions.to_csv(
    "Weather_Predictions.csv",
    index=False
)

print("\nPrediction File Saved Successfully!")