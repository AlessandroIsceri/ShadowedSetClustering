import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from _utils import *

class ShadowedCMeans(BaseEstimator,ClusterMixin):
    def __init__(self, n_clusters = 2, fuzzifier = 2, max_iter = 100, min_distance = 0.5, verbose = True):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.max_iter = max_iter
        self.min_distance = min_distance
        self.verbose = verbose


    def fit(self, X, y = None):
        n_instances = len(X)
        n_features = len(X[0])
        centroids = initialize_random_centroids(X, self.n_clusters, n_features, self.min_distance)
        instances_status = np.zeros((self.n_clusters, n_instances), dtype=int)

        t = 1

        while t < self.max_iter:

            if self.verbose:
                print('iteration number ', t)

            distances = compute_distances(centroids, X)
            u = compute_u(self.n_clusters, n_instances, self.fuzzifier, distances)  #step 3

            thresholds = compute_thresholds(self.n_clusters, n_instances, u)

            last_instances_status = instances_status.copy()
            compute_centroids(centroids, X, u, thresholds, self.fuzzifier, instances_status)

            if(np.array_equal(last_instances_status, instances_status)): #stopping criteria
                break

            t = t + 1

        if self.verbose:
            print("******************" + "*" * len(str(t)))
            for i in range(len(centroids)):
                print("final centroid ", i, " = ", centroids[i])

        output = compute_output(instances_status, n_instances, self.n_clusters)
        return output