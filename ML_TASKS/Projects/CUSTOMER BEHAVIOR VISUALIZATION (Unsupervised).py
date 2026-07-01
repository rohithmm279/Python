# ============================================================
# PROJECT 1 — CUSTOMER BEHAVIOR VISUALIZATION (Unsupervised)
# Dataset: Customer Personality Analysis
# PCA + t-SNE
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Step 1: Load data (tab-separated) ──────────────────────────
df = pd.read_csv("ML_TASKS\\marketing_campaign.csv", sep="\t")

# ── Step 2: Clean ────────────────────────────────────────────
df['Income'] = df['Income'].fillna(df['Income'].median())
df = df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1)

df['Age'] = 2026 - df['Year_Birth']
df = df[df['Age'] < 100]   # remove a few bad birth-year outliers

# ── Step 3: Feature engineering ────────────────────────────────
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spent'] = df[spend_cols].sum(axis=1)

purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Customer_Tenure'] = (pd.to_datetime('2026-01-01') - df['Dt_Customer']).dt.days

df = df.drop(['Year_Birth', 'Dt_Customer'], axis=1)

# ── Step 4: Encode categorical columns ─────────────────────────
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

# ── Step 5: Separate features and coloring variable ────────────
color_var = df['Total_Spent']       # used ONLY to color the plot later — not used in fitting
X_vis = df.drop('Total_Spent', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vis)
print(f"Final feature shape: {X_scaled.shape}")

# ── Step 6: PCA — fit and transform to 2D ───────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA output shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.2%}")

# ── Step 7: Visualize PCA result ────────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=color_var, cmap='viridis', alpha=0.7, s=20
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA — Customer Purchasing Behavior (2D Projection)")
plt.colorbar(scatter, label='Total Amount Spent')
plt.tight_layout()
plt.show()

# ── Step 8: t-SNE — fit and transform to 2D ──────────────────────
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
print(f"t-SNE output shape: {X_tsne.shape}")

# ── Step 9: Visualize t-SNE result ────────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=color_var, cmap='viridis', alpha=0.7, s=20
)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE — Customer Purchasing Behavior (2D Projection)")
plt.colorbar(scatter, label='Total Amount Spent')
plt.tight_layout()
plt.show()