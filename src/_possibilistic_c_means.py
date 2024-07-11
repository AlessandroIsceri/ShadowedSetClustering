#Modified from: https://github.com/holtskinner/PossibilisticCMeans

##
# MIT License

# Copyright (c) 2018 Holt Skinner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##

import numpy as np
from scipy.spatial.distance import cdist
    
def eta(u, d, fuzzifier):
    u = u ** fuzzifier
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)
    return n


def _fcm(instances, centroids, n, fuzzifier, metric="euclidean"):
    d = cdist(instances.T, centroids, metric=metric).T
    d = np.fmax(d, np.finfo(instances.dtype).eps)

    exp = -2. / (fuzzifier - 1)
    d2 = d ** exp
    u = d2 / np.sum(d2, axis=0, keepdims=1)

    return u, d


def _pcm(instances, centroids, n, fuzzifier, metric="euclidean"):
    d = cdist(instances.T, centroids, metric=metric)
    d = np.fmax(d, np.finfo(instances.dtype).eps)

    d2 = (d ** 2) / n
    exp = 1. / (fuzzifier - 1)
    d2 = d2.T ** exp
    u = 1. / (1. + d2)

    return u, d


def _update_clusters(instances, u, fuzzifier):
    um = u ** fuzzifier
    centroids = um.dot(instances.T) / np.atleast_2d(um.sum(axis=1)).T
    return centroids


def possibilistic_c_means(centroids, instances, c, fuzzifier, method, t_max = 100, n = 1, verbose = True):
    # Membership Matrix Each Data Point in each cluster
    u = np.zeros((len(instances), c))

    # Number of Iterations
    t = 0

    metric = "euclidean"

    while t < t_max - 1:

        if method == 'POSSIBILISTIC':
            u, d = _pcm(instances, centroids, n, fuzzifier, metric)
        else:
            u, d = _fcm(instances, centroids, n, fuzzifier, metric)

        old_centroids = centroids.copy()
        centroids = _update_clusters(instances, u, fuzzifier)

        if np.linalg.norm(old_centroids - centroids) < 0.1:
            break

        t = t + 1
    if method == 'POSSIBILISTIC':
        return u
    else:
        return u, d