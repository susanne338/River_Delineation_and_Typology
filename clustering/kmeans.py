import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data----------------------------------------------------------------------------------------------------
df = pd.read_csv('your_river_data.csv')
print(df.head())

# ----------------------------------------------------------------------------------------------------
# Pre-process as clustering algorithms are sensitive to feature scaling, so standardize my features: mean=0. variance=1
# Select only the numeric columns for clustering
numeric_data = df.select_dtypes(include=['float64', 'int64'])
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)


# Choose number of clusters with elbow method.----------------------------------------------------------------
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Apply clustering algorithm ----------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)




# Visualize------------------------------------------------------------------------------------------------
# If more than two dimensions, we need to apply pca to reduce the data to wo principal components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
plt.title('River Spaces Clusters')
plt.show()

