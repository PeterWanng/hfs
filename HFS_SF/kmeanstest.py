print(__doc__)

import time

import numpy as np
import pylab as pl

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs

##############################################################################
# Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

##############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
a=k_means.fit(X)
t_batch = time.time() - t0
k_means_labels = k_means.labels_

print len(k_means_labels)
k_means_cluster_centers = k_means.cluster_centers_
print len(k_means_cluster_centers)
k_means_labels_unique = np.unique(k_means_labels)
