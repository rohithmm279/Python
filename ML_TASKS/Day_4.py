import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    classification_report
)

# Load Dataset
df = pd.read_csv(r"R:\S5 AI roadmap\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# Logistic Results
print("===== LOGISTIC REGRESSION =====")

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("MAE:",
      mean_absolute_error(y_test, lr_pred))

print("MSE:",
      mean_squared_error(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

# Logistic Confusion Matrix
cm_lr = confusion_matrix(y_test, lr_pred)

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d'
)

plt.title("Logistic Regression")
plt.show()

# Random Forest Results
print("===== RANDOM FOREST =====")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("MAE:",
      mean_absolute_error(y_test, rf_pred))

print("MSE:",
      mean_squared_error(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d'
)

plt.title("Random Forest")
plt.show()

# Accuracy Comparison
models = [
    "Logistic",
    "Random Forest"
]

accuracy = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, rf_pred)
]

plt.bar(models, accuracy)

plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()