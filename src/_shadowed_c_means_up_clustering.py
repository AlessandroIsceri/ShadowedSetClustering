import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.optimize import Bounds
from scipy.optimize import minimize

from _utils import *

class ShadowedCMeansUp(BaseEstimator,ClusterMixin):
    def __init__(self, n_clusters = 2, fuzzifier = 2, optimization_type = "none", max_iter = 100, min_distance = 0.5, verbose = True):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.optimization_type = optimization_type
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

            thresholds = self._compute_thresholds(n_instances, u)

            last_instances_status = instances_status.copy()
            compute_centroids(centroids, X, u, thresholds, self.fuzzifier, instances_status)

            if(np.array_equal(last_instances_status, instances_status)):
                break

            t = t + 1

        if self.verbose:
            print("******************" + "*" * len(str(t)))
            for i in range(len(centroids)):
                print("final centroid ", i, " = ", centroids[i])
            
        output = compute_output(instances_status, n_instances, self.n_clusters)
        return output
    
    
    def _compute_thresholds(self, n_instances, u):  #(8)
        thresholds = np.zeros(self.n_clusters)

        for i in range(self.n_clusters):
            u_ik_max = max(u[i])
            u_ik_min = min(u[i])
            bounds = Bounds([0], [(u_ik_max + u_ik_min) / 2])
            x0 = (u_ik_max + u_ik_min) / 3 # initial value
            
            if self.optimization_type == 'none':
                res = minimize(self._base_objective_function, x0, args = (n_instances, i, u), bounds=bounds)
            elif self.optimization_type == 'gradualness':
                res = minimize(self._gradualness_balance_function, x0, args = (n_instances, i, u), bounds=bounds)
            elif self.optimization_type == 'trade-off': #maximize -> return -function
                res = minimize(self._trade_off_balance_function, x0, args = (n_instances, i, u), bounds=bounds)
            elif self.optimization_type == 'sharpness':
                res = minimize(self._sharpness_balance_function, x0, args = (n_instances, i, u), bounds=bounds)
            
            thresholds[i] = res.x[0]
        return thresholds


    def _gradualness_balance_function(self, alpha, n_instances, i, u):
        #i-th centroid
        a = 0
        b = 0
        c = 0
        d = 0
        u_i_max = max(u[i])

        for k in range(n_instances):
            if(u[i, k] >= (u_i_max - alpha)):
                # k-th istance is CORE for this cluster
                c = c + (1 - u[i, k])
            elif(alpha < u[i, k] and u[i, k] < (u_i_max - alpha)):
                # k-th istance is SHADOW for this cluster
                if(0.5 <= u[i, k] and u[i, k] <= 1 - alpha):
                    #superior_shadow
                    b = b + (u[i, k] - 0.5)
                else: #elif(alpha <= u[i, k] and u[i, k] < 0.5):
                    #inferior shadow
                    d = d + (0.5 - u[i, k])
            else:
                # k-th istance is EXCLUSION for this cluster
                a = a + u[i, k]
        return abs(a + b - c -d)


    def _trade_off_balance_function(self, alpha, n_instances, i, u):
        #i-th centroid
        a = 0
        u_i_max = max(u[i])

        for k in range(n_instances):
            if(u[i, k] >= (u_i_max - alpha)):
                # k-th istance is CORE for this cluster
                a = a + u[i, k]
            elif(alpha < u[i, k] and u[i, k] < (u_i_max - alpha)):
                # k-th istance is SHADOW for this cluster
                a = a - u[i, k]
            else:
                # k-th istance is EXCLUSION for this cluster
                a = a + u[i, k]
        return -a


    def _sharp(u, i, k):
        return abs(u[i, k] - 0.5)


    def _sharpness_balance_function(self, alpha, n_instances, i, u):
        #i-th centroid
        g = 0
        h = 0
        f = 0
        u_i_max = max(u[i])

        for k in range(n_instances):

            if(u[i, k] >= (u_i_max - alpha)):
                # k-th istance is CORE for this cluster
                f = f + (0.5 - self._sharp(u, i, k))
            elif(alpha < u[i, k] and u[i, k] < (u_i_max - alpha)):
                # k-th istance is SHADOW for this cluster
                g = g + self._sharp(u, i, k)
            else:
                # k-th istance is EXCLUSION for this cluster
                h = h + (0.5 - self._sharp(u, i, k))
        return abs(g - h - f)