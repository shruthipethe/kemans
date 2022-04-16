import numpy as np
import matplotlib.pyplot as plt
import math
import random

X = np.array([[2, 4],
    [1.7, 2.8],
    [7, 8],
    [8.6, 8],
    [3.4, 1.5],
    [9,11]])

def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def closestCentroid(x, centroids):
    assignments = []
    for i in x:
        # distance between one data point and centroids
        distance=[]
        for j in centroids:
            dist = euclidean_distance(i, j)
            if math.isnan(dist):
                distance.append(random.randint(0,9))
            else:
                distance.append(euclidean_distance(i, j))
            # assign each data point to the cluster with closest centroid
        assignments.append(np.argmin(distance))
    return np.array(assignments)

def updateCentroid(x, clusters, K):
    new_centroids = []
    for c in range(K):
        # Update the cluster centroid with the average of all points 
        # in this cluster
        cluster_mean = x[clusters == c].mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

def kmeans(x, K):
    centroids = [20 * np.random.rand(2) for i in range(K)]
    for i in range(10):
        clusters = closestCentroid(x, centroids)
        centroids = updateCentroid(x, clusters, K)
        print("Centroid:", list(centroids[0]), list(centroids[1]))
    clusters = closestCentroid(x, centroids)
    return clusters, centroids

K = 2
clusters, centroids = kmeans(X, K)
print(clusters)

colors = []
for cluster in clusters:
    if cluster == 0:
        colors.append('red')
    else:
        colors.append('blue')

plt.scatter(X[:,0], X[:,1], s=150, c=colors)
plt.scatter(centroids[0][0], centroids[0][1], c='green')
plt.scatter(centroids[1][0], centroids[1][1], c='yellow')
plt.show()