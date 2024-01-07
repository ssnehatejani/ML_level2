# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:10:47 2024

@author: Sneha
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/Sneha/Downloads/Iris Dataset.csv")

# Store the Species column for future comparison
species = df['Species']

# Remove the Species column
df = df.drop('Species', axis=1)

# Data Preprocessing
# In this example, we'll drop non-numeric columns and handle missing values
df = df.select_dtypes(include=[np.number]).fillna(0)

# K-Means Clustering Implementation
def kmeans_clustering(dataset, k=3, max_iters=100):
    # Randomly initialize centroids
    centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(dataset[:, None] - centroids, axis=-1), axis=-1)
        
        # Update centroids to be the mean of the points assigned to each cluster
        centroids = np.array([dataset[labels == i].mean(axis=0) for i in range(k)])
    
    return labels

# Principal Component Analysis Implementation
def pca(dataset):
    # Center the data
    centered_data = dataset - np.mean(dataset, axis=0)
    
    # Calculate covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Project the data onto the first three principal components
    pca_result = np.dot(centered_data, eigenvectors[:, :3])
    
    return pca_result, eigenvalues

# Example Usage
kmeans_clusters = kmeans_clustering(df.values, k=3)
pca_result, eigenvalues = pca(df.values)

# Visualization (Bonus)
# For K-Means Clustering
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_clusters, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# For PCA
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=kmeans_clusters, cmap='viridis')
ax1.set_title('PCA with K-Means Clustering')
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_zlabel('Principal Component 3')

ax2 = fig.add_subplot(122)
ax2.bar(range(1, len(eigenvalues) + 1), eigenvalues / np.sum(eigenvalues))
ax2.set_title('Explained Variance Ratio')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Variance Ratio')

plt.show()
