import numpy as np
import math
import random
from sklearn.kernel_approximation import RBFSampler #RBF kernel = Gaussian kernel
from sklearn.base import BaseEstimator, ClusterMixin

from _utils import *
from SRFF import *

class MKSCM(BaseEstimator, ClusterMixin):
    
    def __init__(self, n_clusters = 2, fuzzifier = 2, n_random_features = 500, trade_off = 2, max_iter = 100, min_distance = 0.5, verbose = True):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.n_random_features = n_random_features
        self.trade_off = trade_off
        self.max_iter = max_iter
        self.min_distance = min_distance
        self.verbose = verbose
        
        
    def fit(self, X, y = None):
        self.kernel_number = 2
        n_instances = len(X)
        n_features = len(X[0])

        centroids = initialize_random_centroids(X, self.n_clusters, n_features, self.min_distance) #step 1
        weights = np.zeros((self.n_clusters, self.kernel_number))
        self._initialize_random_weights(weights)
        
        instances_status = np.zeros((self.n_clusters, n_instances), dtype=int)
        mapped_centroids = np.zeros((self.kernel_number, self.n_clusters, self.n_random_features))
        mapped_instances = np.zeros((self.kernel_number, n_instances, self.n_random_features))

        srff = self._define_SRFF(X)

        mapped_instances[0] = self._map_gaussian_kernel(X)
        mapped_instances[1] = self._map_polynomial_kernel(X, srff)

        t = 1

        while t < self.max_iter:  #step 6

            if self.verbose:
                print('iteration number ', t)

            mapped_centroids[0] = self._map_gaussian_kernel(centroids)
            mapped_centroids[1] = self._map_polynomial_kernel(centroids, srff)

            distances = self._compute_distances(mapped_centroids, mapped_instances, weights, n_instances)
            u = compute_u(self.n_clusters, n_instances, self.fuzzifier, distances)  #step 2

            thresholds = compute_thresholds(self.n_clusters, n_instances, u) #step 3
            
            last_instances_status = instances_status.copy()
            compute_centroids(centroids, X, u, thresholds, self.fuzzifier, instances_status) #step 4

            weights = self._compute_weights(mapped_centroids, mapped_instances, u)

            if(np.array_equal(last_instances_status, instances_status)): #stopping criteria
                break

            t = t + 1

        if self.verbose:
            print("******************" + "*" * len(str(t)))
            for i in range(len(centroids)):
                print("final centroid ", i, " = ", centroids[i])

        output = compute_output(instances_status, n_instances, self.n_clusters)
        return output
    
    def _map_gaussian_kernel(self, points):
        gamma = 0.05 # gamma parameter for the Gaussian kernel
        rbf_feature = RBFSampler(gamma=gamma, n_components=self.n_random_features, random_state = 10)  #random state is used for reproducibility
        # Transform the data to Fourier feature space
        mapped_points = rbf_feature.fit_transform(points)
        return mapped_points
    

    def _define_SRFF(self, points):
        DIM = points.shape[1]
        EPS, GRID_SIZE = 1e-20, 500
        A = 1
        P = 2

        def polynomial_kernel_scalar(z, a = A, degree = P):
            return (1- (z/a)**2)**degree

        Kapp = KernelApprox(dim = DIM, kernelfunc = polynomial_kernel_scalar, eps = EPS, grid_size=GRID_SIZE, N = 10, verbose = False)
        Kapp.fit(eval_grid_factor=2)
        cdf = Kapp.get_cdf()

        srff = SRFF(cdf=cdf, a = A, p = P, D = self.n_random_features)
        srff.fit(points)
        return srff
    
    
    def _map_polynomial_kernel(self, points, srff):
        mapped_points = srff.transform(points)
        return mapped_points
    
    
    def _initialize_random_weights(self, weights):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i, j] = random.random()
                
                
    def _compute_distances(self, mapped_centroids, mapped_instances, weights, n_instances):
        distances = np.zeros((self.n_clusters, n_instances)) #create distance matrix
        for j in range(self.n_clusters):
            for n in range(n_instances):
                distance = 0
                for k in range(self.kernel_number):
                    internal_sum = 0
                    for m in range(self.n_random_features):
                        internal_sum = internal_sum + ((mapped_instances[k, n, m] - mapped_centroids[k, j, m])**2)
                    distance = distance + internal_sum * weights[j, k]
                distances[j, n] = distance
        return distances
    
    
    def _compute_weights(self, mapped_centroids, mapped_instances, u):
        weights = np.zeros((self.n_clusters, self.kernel_number))
        for j in range(self.n_clusters):
            for k in range(self.kernel_number):
                numerator = math.exp(- self.trade_off * sum(u[j,n] ** self.fuzzifier * sum((mapped_instances[n,k,m] - mapped_centroids[n,k,m])**2 for m in range(self.n_random_features)) for n in range(len(mapped_instances))))
                denominator = sum(math.exp(- self.trade_off * sum(u[j,n] ** self.fuzzifier * sum((mapped_instances[n,h,m] - mapped_centroids[n,h,m])**2 for m in range(self.n_random_features)) for n in range(len(mapped_instances)))) for k in range(self.kernel_number) for h in range(self.kernel_number))
                weights[j, k] = numerator/denominator
        return weights