from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples

trainingData = pd.read_csv('skewedData.csv').values

def kMeans(inputTrain): #this is the normal k-means clustering, but each time clusters with k=2
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0, n_init='auto')
    kmeans.fit(inputTrain)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    # plot(inputTrain, centroids, labels)
    silhouetteAvg = silhouette_score(inputTrain, labels)
    silhouetteVals = silhouette_samples(inputTrain, labels)

    numClusters = len(np.unique(labels))
    silhouetteClusterVals = {}
    for i in range(numClusters):
        thisClusterSilVals = silhouetteVals[labels == i]
        thisClusterSilAvg = np.mean(thisClusterSilVals)
        silhouetteClusterVals[i] = thisClusterSilAvg
        print(f"Cluster {i}: {len(thisClusterSilVals)} samples, avg silhouette = {thisClusterSilAvg:.4f}")
    print(f"Avg: {silhouetteAvg}")
    plot(inputTrain, centroids, labels)


# def clusterSilhouetteScore(centroi)

def plot(inputTrain, centroids, labels): #make a pretty looking plot
    plt.figure(figsize=(10, 6))
    for i in range(2):
        cluster_points = inputTrain[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}', alpha=0.7, s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroid', linewidths=2)
    plt.xlabel('Attribute 1'); plt.ylabel('Attribute 2'); plt.title(f'Clustering Results')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig('silPlot.png')

kMeans(trainingData)