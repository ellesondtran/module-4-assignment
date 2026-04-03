import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('Pokemon_Complete_Gen1_to_Gen9.csv')
print(f"Loaded {len(df)} Pokemon with columns: {df.columns.tolist()}")

# ── Feature Selection ──────────────────────────────────────────────────────
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
X = df[features].copy()

# ── Standardize ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow Method to Select K ──────────────────────────────────────────────
inertias, sil_scores = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0d0d0d')
for ax in axes:
    ax.set_facecolor('#1a1a2e')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#555')
    ax.tick_params(colors='#ccc')

axes[0].plot(list(k_range), inertias, 'o-', color='#ee1515', linewidth=2)
axes[0].axvline(x=5, color='#ffcb05', linestyle='--', alpha=0.7, label='k=5 chosen')
axes[0].set_title('Elbow Method', color='white')
axes[0].set_xlabel('k', color='#ccc')
axes[0].set_ylabel('Inertia', color='#ccc')
axes[0].legend(facecolor='#1a1a2e', labelcolor='white')

axes[1].plot(list(k_range), sil_scores, 'o-', color='#3b4cca', linewidth=2)
axes[1].axvline(x=5, color='#ffcb05', linestyle='--', alpha=0.7, label='k=5 chosen')
axes[1].set_title('Silhouette Score', color='white')
axes[1].set_xlabel('k', color='#ccc')
axes[1].set_ylabel('Score', color='#ccc')
axes[1].legend(facecolor='#1a1a2e', labelcolor='white')

plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

# ── Final Clustering: K=5 ─────────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)
df['total'] = df[features].sum(axis=1)

cluster_labels = {
    0: 'Defensive Walls',
    1: 'Baby/Unevolved',
    2: 'Fast Sweepers',
    3: 'Legendary Titans',
    4: 'Bulky Attackers'
}
cluster_colors = {
    0: '#3b4cca',
    1: '#78c850',
    2: '#ee1515',
    3: '#ffcb05',
    4: '#ff7300'
}

df['archetype'] = df['cluster'].map(cluster_labels)

# ── Cluster Summary ───────────────────────────────────────────────────────
print("\n── Cluster Means ──")
print(df.groupby('archetype')[features + ['total']].mean().round(1))

print("\n── Cluster Sizes ──")
print(df['archetype'].value_counts())

# ── Heatmap ───────────────────────────────────────────────────────────────
cluster_means = df.groupby('cluster')[features].mean().round(1)
cluster_means.index = [cluster_labels[i] for i in cluster_means.index]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#0d0d0d')
data = cluster_means.values
norm = (data - data.min(0)) / (data.max(0) - data.min(0))
im = ax.imshow(norm, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(6))
ax.set_xticklabels(['HP','Attack','Defense','Sp.Atk','Sp.Def','Speed'], color='white')
ax.set_yticks(range(5))
ax.set_yticklabels(cluster_means.index, color='white')
for i in range(5):
    for j in range(6):
        ax.text(j, i, f'{data[i,j]:.0f}', ha='center', va='center',
                color='black', fontweight='bold')
ax.set_title('Average Stats per Cluster', color='white', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax).set_label('Relative Strength', color='#ccc')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

# ── Scatter: Speed vs Attack ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d0d0d')
ax.set_facecolor('#1a1a2e')
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom','left']:
    ax.spines[spine].set_color('#555')
ax.tick_params(colors='#ccc')

for c in range(5):
    sub = df[df['cluster'] == c]
    ax.scatter(sub['speed'], sub['attack'], color=cluster_colors[c],
               alpha=0.5, s=25, label=cluster_labels[c])

ax.set_xlabel('Speed', color='#ccc', fontsize=12)
ax.set_ylabel('Attack', color='#ccc', fontsize=12)
ax.set_title('Pokémon Clusters: Speed vs Attack', color='white', fontsize=14, fontweight='bold')
ax.legend(facecolor='#1a1a2e', labelcolor='white')
ax.grid(color='#333', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close()

print("\nAll plots saved! Check elbow_plot.png, heatmap.png, scatter_plot.png")
