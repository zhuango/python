#! /usr/bin/python3
#For the simple task of finding the nearest neighbors between two sets of data, the unsupervised algorithms within sklearn.neighbors can be used:
from sklearn.neighbors import NearestNeighbors
import numpy as np
#                 0        1         2       3       4       5
X = np.array([[-1, -1], [-2,-1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)
print(distances)

nearestConnectionMatrix = nbrs.kneighbors_graph(X).toarray()
print(nearestConnectionMatrix)

# use KD-tree or Ball-tree
from sklearn.neighbors import KDTree
import numpy as np
kdt = KDTree(X, leaf_size=30, metric='euclidean')
result = kdt.query(X, k = 2, return_distance=False)
print(result)