# ============================================================
# PROJECT 2 — FRAUD DETECTION (Supervised Learning)
# 3 Models: Logistic Regression, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ── Step 1: Load data ────────────────────────────────────────
df = pd.read_csv("ML_TASKS\\creditcard.csv")

print(f"Shape     : {df.shape}")
print(f"Any nulls : {df.isnull().sum().any()}")

# ── Step 2: Class distribution ───────────────────────────────
counts = df['Class'].value_counts()
pct = df['Class'].value_counts(normalize=True) * 100
print(f"\nLegitimate : {counts[0]:,}  →  {pct[0]:.2f}%")
print(f"Fraud      : {counts[1]:,}  →  {pct[1]:.2f}%")

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, hue='Class', palette=['#2a78d6', '#e34948'], legend=False)
plt.xticks([0, 1], ['Legitimate', 'Fraud'])
plt.title("Class Distribution — Massive Imbalance")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ── Step 3: Scale Amount & Time ──────────────────────────────
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

# ── Step 4: Train/test split ─────────────────────────────────
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Fraud in train: {y_train.sum()}")
print(f"Test shape : {X_test.shape}, Fraud in test : {y_test.sum()}")

# ── Helper: reusable evaluation function ─────────────────────
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} Results ===")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))
    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    return auc

results = {}

# ── Model 1: Logistic Regression ─────────────────────────────
log_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)

# ── Model 2: Random Forest ───────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
results['Random Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# ── Model 3: Decision Tree ───────────────────────────────────
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=8,                                  # limits tree depth to reduce overfitting
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
results['Decision Tree'] = evaluate_model("Decision Tree", y_test, y_pred_dt, y_prob_dt)



# ── Final Comparison ─────────────────────────────────────────
print("\n" + "=" * 40)
print("MODEL COMPARISON (ROC-AUC)")
print("=" * 40)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name:<25}: {auc:.4f}")