# ==========================================
# CREDIT CARD CUSTOMER SEGMENTATION
# K-MEANS CLUSTERING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(r'R:\S5 AI roadmap\CC GENERAL.csv')

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================
# DATA PREPROCESSING
# ==========================================

# Remove Customer ID
if 'CUST_ID' in df.columns:
    df.drop('CUST_ID', axis=1, inplace=True)

# Fill Missing Values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# Standardize Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_imputed)

# ==========================================
# ELBOW METHOD
# ==========================================

wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(scaled_data)
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ==========================================
# SILHOUETTE SCORE
# ==========================================

print("\nSilhouette Scores")

for k in range(2,8):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    score = silhouette_score(
        scaled_data,
        labels
    )

    print(f"K={k} : {score:.4f}")

# ==========================================
# TRAIN FINAL MODEL
# ==========================================

optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df_imputed["Cluster"] = clusters

# ==========================================
# CLUSTER COUNTS
# ==========================================

print("\nCluster Distribution")
print(df_imputed["Cluster"].value_counts())

# ==========================================
# CLUSTER SUMMARY
# ==========================================

cluster_summary = df_imputed.groupby("Cluster").mean()

print("\nCluster Summary")
print(cluster_summary)

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(8,5))
sns.countplot(
    x='Cluster',
    data=df_imputed
)
plt.title("Customers per Cluster")
plt.show()

# ==========================================
# HEATMAP
# ==========================================

plt.figure(figsize=(12,6))

sns.heatmap(
    cluster_summary,
    cmap="coolwarm",
    annot=False
)

plt.title("Cluster Feature Means")
plt.show()

# ==========================================
# SAVE RESULTS
# ==========================================

df_imputed.to_csv(
    "Credit_Card_Segmentation_Output.csv",
    index=False
)

print("\nResults Saved Successfully!")