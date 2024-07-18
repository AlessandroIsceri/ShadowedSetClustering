# Shadowed Set Clustering Algorithms

## Description

This project serves as the internal stage at the University of Milan Bicocca for my bachelor's degree thesis in Computer Science. The thesis is based on the development, implementation, and comparison of five shadowed set clustering algorithms in Python.

This project is an extension of the scikit-learn library.

The five algorithms implemented are:
 - [Shadowed C-Means](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/src/_shadowed_c_means_clustering.py)
 - [Shadowed C-Means Upgraded](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/src/_shadowed_c_means_up_clustering.py)
 - [Shadowed Weighted C-Means](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/src/_shadowed_weighted_c_means_clustering.py)
 - [Multiple Kernel Shadowed C-Means](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/src/_multiple_kernel_shadowed_c_means.py)
 - [Shadowed Set-Based Multi-Granular Three-Way Clustering Ensemble](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/src/_SM3WCE.py)

The algorithms were implemented based on several research papers.

## Installation

No installation is needed. Simply copy the files and use the classes provided.

## Usage

An example usage of the Shadowed C-Means algorithm on the Iris dataset:

```
# import useful libraries
import numpy as np
from sklearn.datasets import load_iris

# import the algorithm
from _shadowed_c_means_clustering import *

# load the dataset
iris = load_iris()
instances = np.array(iris.data) #numpy array is preferred

# use the SCM algorithm on the Iris dataset 
SCM = ShadowedCMeans(3) #3 clusters
SCM.fit(instances)

```

## Algotithms and parameters

### ShadowedCMeans

`class ShadowedCMeans(n_clusters = 2, fuzzifier = 2, max_iter = 100, min_distance = 0.5, verbose = True)`

**Parameters:**

 - `n_clusters : int, default = 2` \
   The number of clusters to form as well as the number of centroids to generate.

 - `fuzzifier : float, default = 2` \
   The value of the fuzzifier used to compute membership grades.

 - `max_iter : int, default = 100` \
   Maximum number of iterations of the SCM algorithm for a single run.

 - `min_distance : float, default = 0.5` \
   Specifies the minimum distance required between initial centroids during the initialization phase of the algorithm.

 - `verbose : boolean, default = True` \
   Verbosity mode.

### ‎ShadowedCMeansUp‎

`class ‎ShadowedCMeansUp‎(n_clusters = 2, fuzzifier = 2, optimization_type = "none", max_iter = 100, min_distance = 0.5, verbose = True)`

**Parameters:**

- `n_clusters : int, default = 2` \
  The number of clusters to form as well as the number of centroids to generate.

 - `fuzzifier : float, default = 2` \
   The value of the fuzzifier used to compute membership grades.

 - `optimization_type‎ : String, default = "none"` \
   Specifies the type of optimization to be employed. Possible values:
   1. `gradualness`: balances the number of instances where the membership grade increases with those where it decreases.
   2. `trade-off`: promotes the definite classification of instances (core/exclusion) rather than the ambiguous one (shadow).
   3. `sharpness`: balances the sharpness changes during the iterations.

 - `max_iter : int, default = 100` \
   Maximum number of iterations of the SCM algorithm for a single run.

 - `min_distance : float, default = 0.5` \
   Specifies the minimum distance required between initial centroids during the initialization phase of the algorithm.

 - `verbose : boolean, default = True` \
   Verbosity mode.

### ShadowedWeightedCMeans‎

`class ShadowedWeightedCMeans‎(n_clusters = 2, fuzzifier = 2, alpha = 8, max_iter = 100, min_distance = 0.5, verbose = True)`

**Parameters:**

 - `n_clusters : int, default = 2` \
   The number of clusters to form as well as the number of centroids to generate.

 - `fuzzifier : float, default = 2` \
   The value of the fuzzifier used to compute membership grades.

 - `alpha : float, default = 8` \
   A useful value used to assign importance to the weights.

 - `max_iter : int, default = 100` \
   Maximum number of iterations of the SCM algorithm for a single run.

 - `min_distance : float, default = 0.5` \
   Specifies the minimum distance required between initial centroids during the initialization phase of the algorithm.

 - `verbose : boolean, default = True` \
   Verbosity mode.

### MKSCM

`class MKSCM(n_clusters = 2, fuzzifier = 2, n_random_features = 500, trade_off = 2, max_iter = 100, min_distance = 0.5, verbose = True)`

**Parameters:**

 - `n_clusters : int, default = 2` \
   The number of clusters to form as well as the number of centroids to generate.

 - `fuzzifier : float, default = 2` \
   The value of the fuzzifier used to compute membership grades.

 - `n_random_features : int, default = 500` \
   The number of features of the newly mapped points.

 - `trade_off : float, default = 2` \
   A parameter used to assign weights to the kernels.

 - `max_iter : int, default = 100` \
   Maximum number of iterations of the SCM algorithm for a single run.

 - `min_distance : float, default = 0.5` \
   Specifies the minimum distance required between initial centroids during the initialization phase of the algorithm.

 - `verbose : boolean, default = True` \
   Verbosity mode.

### SM3WCE‎

`class SM3WCE‎(n_clusters = 2, fuzzifier = 2, m = 5, max_iter = 100, min_distance = 0.5, verbose = True)`

**Parameters:**

 - `n_clusters : int, default = 2` \
   The number of clusters to form as well as the number of centroids to generate.
   
 - `fuzzifier : float, default = 2` \
   The value of the fuzzifier used to compute membership grades.

 - `m : int, default = 5` \
   The number of times the PCM algorithm is executed.

 - `max_iter : int, default = 100` \
   Maximum number of iterations of the SCM algorithm for a single run.

 - `min_distance : float, default = 0.5` \
   Specifies the minimum distance required between initial centroids during the initialization phase of the algorithm.

 - `verbose : boolean, default = True` \
   Verbosity mode.

## Output (for all algorithms)

Every implemented algorithm returns a matrix with n (#instances) rows and c+1 (#clusters + 1) columns.

The i-th row of the matrix represent the "state" of the i-th instance.

### Core element for one cluster

If the i-th instance is a core element for the j-th cluster, then the i-th row of the returned matrix will be:

                          j             c
    i-th row: [ 0 0 ... 0 1 0 ... 0 0 | 0 ]

Note: c (instead of c+1) because indexing in Python starts from 0!
    
### Shadow element for one cluster

If the i-th instance is a shadow element ONLY for the j-th cluster, then the i-th row of the returned matrix will be:

                          j             c
    i-th row: [ 0 0 ... 0 1 0 ... 0 0 | 1]

### Shadow element for more than one cluster

If the i-th instance is a shadow element for the j-th and the k-th cluster, then the i-th row of the returned matrix will be:

                          j        k            c
    i-th row: [ 0 0 ... 0 1 0 ... 0 1 0 ... 0 | 0]

### Exclusion element for all clusters

If the i-th instance is excluded from all clusters, then the i-th row of the returned matrix will be:

                              c
    i-th row: [ 1 1 ... 1 1 | 1]

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/AlessandroIsceri/ShadowedSetClustering/blob/main/LICENSE.md) file for details.

## References

 - Sushmita Mitra, Witold Pedrycz, and Bishal Barman. Shadowed c-means: Integrating fuzzy and rough clustering. Pattern Recognition, 43(4):1282 – 1291, 2010.
 - Tamunokuro William-West, Armand Florentin Donfack Kana, and Musa Adeku Ibrahim. Shadowedset-based three-way clustering methods: An investigation of new optimization-based principles. Information Sciences, 591:1 – 24, 2022.
 - Lina Wang, Jiandong Wang, and Jian Jiang. New shadowed c-means clustering with feature weights. Transactions of Nanjing University of Aeronautics and Astronautics, 29(3):273 – 283, 2012.
 - Yin-Ping Zhao, Long Chen, and C. L. Philip Chen. Multiple kernel shadowed clustering in approximated feature space. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 10943 LNCS:265 – 275, 2018.
 - ChunMao Jiang, ZhiCong Li, and JingTao Yao. A shadowed set-based three-way clustering ensemble approach. International Journal of Machine Learning and Cybernetics, 13(9):2545 – 2558, 2022.





