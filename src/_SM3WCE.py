import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.optimize import Bounds
from scipy.optimize import minimize

from _possibilistic_c_means import *
from _utils import *

class SM3WCE(BaseEstimator,ClusterMixin):
    def __init__(self, n_clusters = 2, fuzzifier = 2, m = 5, max_iter = 100, min_distance = 0.5, verbose = True):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.m = m
        self.max_iter = max_iter
        self.min_distance = min_distance
        self.verbose = verbose
    
    
    def fit(self, X, y = None):
        X = np.array(X, dtype = np.float64) #convert to the right type
        n_instances = len(X)
        n_features = len(X[0])
        membership_grades = np.zeros((self.m, self.n_clusters, n_instances))
        centroids = initialize_random_centroids(X, self.n_clusters, n_features, self.min_distance)

        for k in range(self.m):
            #execute possibilistic c-means m times
            if self.verbose:
                print("iteration number ", (k+ 1))
            
            for i in range(self.n_clusters):
                if k % 2 == 0:
                    centroids[i] = centroids[i] + self._find_nearest(centroids[i], X) * (np.random.rand() * 0.01)
                else:
                    centroids[i] = centroids[i] - self._find_nearest(centroids[i], X) * (np.random.rand() * 0.01)
            u, d = possibilistic_c_means(centroids, X.T, self.n_clusters, self.fuzzifier, 'FCM', self.max_iter)
            eta_var = eta(u, d, self.fuzzifier)
            membership_grades[k] = possibilistic_c_means(centroids, X.T, self.n_clusters, self.fuzzifier, 'POSSIBILISTIC', self.max_iter, eta_var)

        for j in range(self.m):
            #iterate through the clusters of πj
            u = membership_grades[j]
            for i in range(self.n_clusters):
                alpha, beta = self._compute_thresholds(u, i)
                for k in range(len(X)):
                    if u[i, k] >= alpha:
                        membership_grades[j, i, k] = 1
                    elif u[i, k] <= beta:
                        membership_grades[j, i, k] = 0

        cores = np.array([set() for _ in range(self.n_clusters)])
        fringes = np.array([set() for _ in range(3 * self.n_clusters)]).reshape(3, self.n_clusters)

        for j in range(self.n_clusters):
            #j-th cluster
            pessimistic_lower_approx_set = self._compute_pessimistic_lower_approx_set(X, membership_grades, j)
            pessimistic_upper_approx_set = self._compute_pessimistic_upper_approx_set(X, membership_grades, j)
            optimistic_lower_approx_set = self._compute_optimistic_lower_approx_set(X, membership_grades, j)
            optimistic_upper_approx_set = self._compute_optimistic_upper_approx_set(X, membership_grades, j)

            cores[j] = pessimistic_lower_approx_set
            fringes[0][j] = optimistic_lower_approx_set - pessimistic_lower_approx_set
            fringes[1][j] = optimistic_upper_approx_set - optimistic_lower_approx_set
            fringes[2][j] = pessimistic_upper_approx_set - optimistic_upper_approx_set

            for i in range(len(fringes)): #Fringe1, Fringe2, Fringe3
                if len(cores[j]) > 0:
                    centroid = np.array([self._compute_centroid(cores[j])])
                else:
                    centroid = centroids[j]
                if i == 2:
                    centroids[j] = centroid

                current_fringe = np.array(list(fringes[i][j]))
                
                distances = self._compute_distances(centroid, current_fringe)
                u = self._compute_u(current_fringe, distances)

                alpha, beta = self._compute_thresholds(u, 0)

                if u.size != 0:
                    for k in range(len(current_fringe)):
                        if u[0, k] >= alpha:
                            cores[j].add(tuple(current_fringe[k]))
                            u[0, k] = 1

                # update fringes
                if i != 2:
                    temp_fringe = fringes[i][j] - cores[j]
                    fringes[i + 1][j] = temp_fringe.union(fringes[i + 1][j])

        instances_status = self._compute_istance_status(cores, fringes[2], X, n_instances)
        
        if self.verbose:
            print("******************" + "*" * len(str(self.m)))
            for i in range(len(centroids)):
                print("final centroid ", i, " = ", centroids[i])
        
        output = compute_output(instances_status, n_instances, self.n_clusters)
        return output
            
    
    def _objective_function(self, params, u, i):
        alpha, beta = params
        a = 0
        b = 0
        c = 0
        d = 0
        
        for k in range(len(u[i])):
            if u[i, k] >= alpha:
                a = a + (1 - u[i, k])
            elif u[i, k] <= beta:
                b = b + u[i, k]
            else:
                c = c + (1 - u[i, k])
                d = d + u[i, k]
        return abs(a + b - (c + d))
    
    
    def _compute_thresholds(self, u, i):
        initial_guess = [0.6, 0.3]
        bounds = Bounds([0, 0], [1, 1])
        result = minimize(self._objective_function, initial_guess, args = (u, i), bounds = bounds, method='Powell')
        a, b = result.x
        return a, b
    
    
    def _compute_pessimistic_lower_approx_set(self, instances, membership_grades, j):
        #j-th cluster
        ret = set()

        for k in range(len(instances)):
            #k-th istance
            count = 0
            for i in range(len(membership_grades)):
                #πi clustering
                u = membership_grades[i]
                if u[j, k] == 1:
                    count = count + 1
            if count == len(membership_grades):
                ret.add(tuple(instances[k]))
        return ret
    
    
    def _compute_pessimistic_upper_approx_set(self, instances, membership_grades, j):
        #j-th cluster
        ret = set()

        for k in range(len(instances)):
            #k-th istance
            count = 0
            for i in range(len(membership_grades)):
                #πi clustering
                u = membership_grades[i]
                if u[j, k] == 0:
                    count = count + 1
            if count == len(membership_grades):
                ret.add(tuple(instances[k]))
        return set(tuple(i) for i in instances) - ret #U - ret
    
    
    def _compute_optimistic_lower_approx_set(self, instances, membership_grades, j):
        #j-th cluster
        ret = set()

        for k in range(len(instances)):
            #k-th istance
            for i in range(len(membership_grades)):
                #πi clustering
                u = membership_grades[i]
                if u[j, k] == 1:
                    ret.add(tuple(instances[k]))
                    break
        return ret
    
    
    def _compute_optimistic_upper_approx_set(self, instances, membership_grades, j):
        #j-th cluster
        ret = set()

        for k in range(len(instances)):
            #k-th istance
            for i in range(len(membership_grades)):
                #πi clustering
                u = membership_grades[i]
                if u[j, k] == 0:
                    ret.add(tuple(instances[k]))
                    break
        return set(tuple(i) for i in instances) - ret #U - ret
    
    
    def _compute_centroid(self, core_set):
        sum = np.zeros(len(next(iter(core_set))))
        for istance in core_set:
            sum = sum + istance
        sum = sum / len(core_set)
        return sum
    
    
    def _compute_distances(self, centroid, instances):
        distances = np.zeros((1, len(instances))) #distances between each object in the fringe and centroid
        for k in range(len(instances)):
            distances[0, k] = (np.linalg.norm(instances[k] - centroid))**2 #euclidean distance
        return distances
    
    
    def _compute_u(self, instances, distances): #trying to compute uik (4),
        u = np.zeros((1, len(instances))) #create u array,
        for i in range(len(instances)):
            u[0, i] = 1 / sum((distances[0, i] / distances[0, j]) ** (2 / (self.fuzzifier - 1)) for j in range(len(instances)))
        return u
    
    
    def _compute_istance_status(self, core_points, shadow_points, instances, n_instances):
        instances_status = np.zeros((self.n_clusters, n_instances), dtype=int)
        for j in range(self.n_clusters):
            for k in range(n_instances):
                if tuple(instances[k]) in core_points[j]:
                    instances_status[j, k] = 2 #'CORE'
                elif tuple(instances[k]) in shadow_points[j]:
                    instances_status[j, k] = 1 #'SHADOW'
                else:
                    instances_status[j, k] = 0 #'EXCLUSION'
        return instances_status
    
    
    def _find_nearest(self, centroid, instances):
        distances = np.zeros(len(instances))
        for k in range(len(instances)):
            distances[k] = (np.linalg.norm(instances[k] - centroid))**2

        random_index = int(np.random.rand() * 3)

        sorted_distances = distances.copy()
        sorted_distances.sort()
        min_dist = sorted_distances[random_index]

        k = np.where(distances == min_dist)
        if len(k[0]) > 1:
            k = k[0][0]
        return instances[k]