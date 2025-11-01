import pandas as pd
import numpy as np
from trading_sentiment_analysis import TradingSentimentAnalyzer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
analyzer = TradingSentimentAnalyzer()
analyzer.load_data('historical_data.csv', 'fear_greed_index.csv')
analyzer.preprocess_data()
analyzer.merge_datasets()

# Select features for pattern recognition (trader behavior)
features = [
    'Closed PnL', 'Size USD', 'Fee', 'trade_hour', 'is_profitable', 'abs_pnl'
]
data = analyzer.merged_df[features].copy()

# Fill missing values if any
for col in features:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Find optimal number of clusters (elbow method)
inertia = []
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Fit KMeans with optimal k (e.g., 3)
k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['cluster'] = clusters

# Visualize clusters (pairplot)
sns.pairplot(data, hue='cluster', palette='husl', diag_kind='kde')
plt.suptitle('Trade Pattern Clusters', y=1.02)
plt.show()

# Show cluster centers (in original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=features)
print('Cluster Centers:')
print(centers_df)

# Save cluster assignments
analyzer.merged_df['pattern_cluster'] = clusters
analyzer.merged_df.to_csv('trading_patterns_with_clusters.csv', index=False)
print('Cluster assignments saved to trading_patterns_with_clusters.csv')
