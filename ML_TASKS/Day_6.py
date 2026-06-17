import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# ---------------------
# ELBOW METHOD
# ---------------------

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()

# ---------------------
# PCA
# ---------------------

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

# ---------------------
# KMEANS AFTER PCA
# ---------------------

kmeans_pca = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters_pca = kmeans_pca.fit_predict(
    X_pca
)

# ---------------------
# PCA VISUALIZATION
# ---------------------

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters_pca
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Customer Segmentation Using PCA")

plt.show()