import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/Mall_Customers.csv')

# Pre-process data
df['Gender'] = df['Gender'].astype('category')
df['GenderCode'] = df['Gender'].cat.codes
df_pp = df.drop(['CustomerID', 'Gender'], axis=1)

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pp)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['KMeans Cluster'] = kmeans.fit_predict(df_scaled)

# Apply Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
df['Hierarchical Cluster'] = hierarchical.fit_predict(df_scaled)

print(df)

# Visualize using Scatter Plots
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['KMeans Cluster'], cmap='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Hierarchical Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering')

plt.show()

# Visualize using Pair Plot
df_pairplot = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'KMeans Cluster', 'Hierarchical Cluster']]
sns.pairplot(df_pairplot, hue='KMeans Cluster', palette='viridis', markers='o', diag_kind='kde')
plt.show()

sns.pairplot(df_pairplot, hue='Hierarchical Cluster', palette='viridis', markers='o', diag_kind='kde')
plt.show()

# Visualize using 3D Scatter Plot
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['KMeans Cluster'], cmap='viridis')
ax1.set_xlabel('Age')
ax1.set_ylabel('Annual Income (k$)')
ax1.set_zlabel('Spending Score (1-100)')
ax1.set_title('K-Means Clustering')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Hierarchical Cluster'],
            cmap='viridis')
ax2.set_xlabel('Age')
ax2.set_ylabel('Annual Income (k$)')
ax2.set_zlabel('Spending Score (1-100)')
ax2.set_title('Hierarchical Clustering')

plt.show()

# Evaluate using Silhouette Score
silhouette_kmeans = silhouette_score(df_scaled, df['KMeans Cluster'])
silhouette_hierarchical = silhouette_score(df_scaled, df['Hierarchical Cluster'])

print(f"Silhouette Score - K-Means: {silhouette_kmeans}")
print(f"Silhouette Score - Hierarchical: {silhouette_hierarchical}")
