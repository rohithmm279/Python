# ============================================================
# PROJECT 2 — CUSTOMER BEHAVIOR VISUALIZATION (Unsupervised)
# PCA + t-SNE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── Step 1: Load data ────────────────────────────────────────
df = pd.read_csv("ML_TASKS\\creditcard.csv")

# ── Step 2: Scale Amount & Time (same as Project 1) ──────────
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

# ── Step 3: Stratified sample (10,000 rows) ───────────────────
# Keep the same fraud ratio as the full dataset in our sample
from sklearn.model_selection import train_test_split

df_sample, _ = train_test_split(
    df, train_size=10000, stratify=df['Class'], random_state=42
)

print(f"Sample shape: {df_sample.shape}")
print(f"Fraud in sample: {df_sample['Class'].sum()}")

# ── Step 4: Separate features and label ────────────────────────
X_vis = df_sample.drop('Class', axis=1)
y_vis = df_sample['Class']   # we keep this ONLY to color the plot later — not used in fitting

# ── Step 5: PCA — fit and transform to 2D ─────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_vis)

print(f"PCA output shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.2%}")

# ── Step 6: Visualize PCA result ───────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y_vis, cmap='coolwarm', alpha=0.6, s=15
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA — Customer Transactions (2D Projection)")
plt.colorbar(scatter, label='Class (0=Legit, 1=Fraud)')
plt.tight_layout()
plt.show()

# ── Step 7: t-SNE — fit and transform to 2D ───────────────────
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_vis)

print(f"t-SNE output shape: {X_tsne.shape}")

# ── Step 8: Visualize t-SNE result ─────────────────────────────
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=y_vis, cmap='coolwarm', alpha=0.6, s=15
)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE — Customer Transactions (2D Projection)")
plt.colorbar(scatter, label='Class (0=Legit, 1=Fraud)')
plt.tight_layout()
plt.show()