import numpy as np
from random import randrange
from scipy.optimize import Bounds
from scipy.optimize import minimize

def compute_distances(centroids, instances):
    distances = np.zeros((len(centroids), len(instances))) #create distance matrix
    for i in range(len(centroids)):
        for k in range(len(instances)):
            distances[i, k] = (np.linalg.norm(instances[k] - centroids[i]))**2 #euclidean distance
    return distances


def compute_u(n_clusters, n_instances, fuzzifier, distances): #compute uik
    u = np.zeros((n_clusters, n_instances)) #create u matrix
    for i in range(n_clusters):
        for k in range(n_instances):
            u[i, k] = 1 / sum((distances[i, k] / distances[j, k]) ** (2 / (fuzzifier - 1)) for j in range(n_clusters))
    return u


def compute_centroids(centroids, instances, u, thresholds, fuzzifier, instances_status):
    for i in range(len(centroids)):
        numerator = 0
        phi = 0
        eta = 0
        psi = 0
        u_i_max = max(u[i])
        
        for k in range(len(instances)):
            if(u[i, k] >= (u_i_max - thresholds[i])):
                numerator = numerator + instances[k]
                phi = phi + 1
                instances_status[i, k] = 2 # 2 = core
            elif(thresholds[i] < u[i, k] and u[i, k] < (u_i_max - thresholds[i])):
                tmp = u[i, k] ** fuzzifier
                numerator = numerator + (tmp * instances[k])
                eta = eta + tmp
                instances_status[i, k] = 1 # 1 = shadow
            else:
                tmp = u[i, k] ** fuzzifier ** fuzzifier
                numerator = numerator + (tmp * instances[k])
                psi = psi + tmp
                instances_status[i, k] = 0 # 0 = exclusion

        centroids[i] = numerator / (phi + eta + psi)


def compute_thresholds(n_clusters, n_instances, u):
    thresholds = np.zeros(n_clusters)

    for i in range(n_clusters):
        u_min = min(u[i])
        u_max = max(u[i])
        bounds = Bounds([u_min], [(u_min + u_max) / 2])
        x0 = (u_min + (u_min + u_max) / 2) / 2
        res = minimize(base_objective_function, x0, args = (n_instances, i, u), bounds=bounds)
        thresholds[i] = res.x[0]
    return thresholds


def base_objective_function(x, n_instances, i, u):
    k = 0
    u_i_max = max(u[i])
    cardinality = 0
    a = 0
    b = 0

    for k in range(n_instances):
        if(x < u[i,k] and u[i,k] < (u_i_max - x)):
            cardinality = cardinality + 1
        elif(u[i, k] <= x):
            a = a + u[i, k]
        elif (u[i, k] >= u_i_max - x):
            b = b + (u_i_max - x)
    return abs(a + b - cardinality)


def initialize_random_centroids(instances, n_clusters, n_features, min_distance):
    finished = False
    centroids = np.zeros((n_clusters, n_features))
    while not finished:

        for i in range(n_clusters):
            index = randrange(len(instances))
            centroids[i] = instances[index] * 0.999

        count = 0
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    if np.linalg.norm(centroids[i] - centroids[j]) < min_distance: #if two initial centroids are too close, re-extract initial centroids
                        count = count + 1
                        break

        if count == 0:
            finished = True
    return centroids


def compute_output(instances_status, n_instances, n_clusters):
    output = np.zeros((n_instances, n_clusters + 1))
    for k in range(n_instances):
        count_shadow = 0
        count_exclusion = 0
        for i in range(n_clusters):
            if instances_status[i, k] == 2: # 2 = CORE
                output[k, i] = 1
            elif instances_status[i, k] == 1: #1 = SHADOW:
                output[k, i] = 1
                count_shadow = count_shadow + 1
            else:
                output[k, i] = 0
                count_exclusion = count_exclusion + 1
        if count_exclusion == n_clusters:  #if the k-th istance is excluded from every cluster -> all ones
            output[k, :] = 1
        if count_shadow == 1:
            output[k, n_clusters] = 1 #last coloumn = 1 -> the point is note core but shadow
    return output