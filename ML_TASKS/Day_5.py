import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv(
    r"R:\S5 AI roadmap\Mall_Customers.csv"
)

# Features
X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Visualization
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=clusters
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation")

plt.show()