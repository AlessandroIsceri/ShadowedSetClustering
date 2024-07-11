import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from _utils import *

class ShadowedWeightedCMeans(BaseEstimator,ClusterMixin):
    def __init__(self, n_clusters = 2, fuzzifier = 2, alpha = 8, max_iter = 100, min_distance = 0.5, verbose = True):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_distance = min_distance
        self.verbose = verbose
        
    
    def fit(self, X, y = None):
        n_instances = len(X)
        n_features = len(X[0])
        u = np.random.rand(self.n_clusters, n_instances) #create u matrix step1
        centroids = np.zeros((self.n_clusters, n_features))
        instances_status = np.zeros((self.n_clusters, n_instances), dtype=int)

        t = 1

        while t < self.max_iter:

            if self.verbose:
                print('iteration number ', t)

            thresholds = compute_thresholds(self.n_clusters, n_instances, u) #step 2
            
            #step 3: according to the threshold determine the lower bound and boundary region for each cluster
            last_instances_status = instances_status.copy()
            compute_centroids(centroids, X, u, thresholds, self.fuzzifier, instances_status) #step 4
            
            distances = compute_distances(centroids, X)
            
            D = self._compute_D(u, distances, n_instances, n_features)
            
            weights = self._compute_weights(D, n_features) #step 5
            
            u = self._compute_u(weights, distances, n_instances, n_features)
            
            if(np.array_equal(last_instances_status, instances_status)): #stopping criteria
                break

            t = t + 1

        if self.verbose:
            print("******************" + "*" * len(str(t)))
            for i in range(len(centroids)):
                print("final centroid ", i, " = ", centroids[i])

        output = compute_output(instances_status, n_instances, self.n_clusters)
        return output
    
    
    def _compute_u(self, weights, distances, n_instances, n_features): #compute uik (4)
        u = np.zeros((self.n_clusters, n_instances)) #create u matrix
        for i in range(self.n_clusters):
            for j in range(n_instances):
                summation = 0
                for t in range(self.n_clusters):
                    num = 0
                    den = 0
                    for k in range(n_features):
                        tmp = weights[k] ** self.alpha
                        num = num + ((tmp) * (distances[i, j]))
                        den = den + ((tmp) * (distances[t, j]))
                    summation = summation + num / den
                summation = (summation ** (2 / (self.fuzzifier - 1))) ** (-1)
                u[i, j] = summation
        return u


    def _compute_D(self, u, distances, n_instances, n_features):
        D = np.zeros(n_features)
        for k in range(n_features):
            for i in range(self.n_clusters):
                for j in range(n_instances):
                    D[k] = (u[i, j] ** self.fuzzifier) * (distances[i, j])
        return D
    
    
    def _compute_weights(self, D, n_features):
        weights = np.zeros(n_features)
        for k in range(n_features):
            weights[k] = 1 / (sum((D[k] / D[t]) ** (1 / (self.alpha - 1)) for t in range(n_features)))
        return weights