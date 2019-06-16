# -*- coding: utf-8 -*-
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
 
def hierarchy_cluster(data, method='average', threshold=5.0):
    data = np.array(data)
 
    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    print(type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)
 
    return num_clusters, indices
 
 
 
def get_cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])
    
    return indices
 
 
if __name__ == '__main__':
    
 
    arr = [[0., 21.6, 22.6, 63.9, 65.1, 17.7, 99.2],
    [21.6, 0., 1., 42.3, 43.5, 3.9, 77.6],
    [22.6, 1., 0, 41.3, 42.5, 4.9, 76.6],
    [63.9, 42.3, 41.3, 0., 1.2, 46.2, 35.3],
    [65.1, 43.5, 42.5, 1.2, 0., 47.4, 34.1],
    [17.7, 3.9, 4.9, 46.2, 47.4, 0, 81.5],
    [99.2, 77.6, 76.6, 35.3, 34.1, 81.5, 0.]]
 
    arr = np.array(arr)
    r, c = arr.shape
    for i in range(r):
        for j in range(i, c):
            if arr[i][j] != arr[j][i]:
                arr[i][j] = arr[j][i]
    for i in range(r):
        for j in range(i, c):
            if arr[i][j] != arr[j][i]:
                print(arr[i][j], arr[j][i])
 
    num_clusters, indices = hierarchy_cluster(arr)
 
 
    print ("%d clusters" % num_clusters)
    for k, ind in enumerate(indices):
        print ("cluster", k + 1, "is", ind)
        
  result:
  <class 'numpy.ndarray'>
5 clusters
cluster 1 is [1 2]
cluster 2 is [5]
cluster 3 is [0]
cluster 4 is [3 4]
cluster 5 is [6]
/Users/xiaofeiwu/anaconda3/envs/tensorflow/lib/python3.5/site-packages/ipykernel_launcher.py:8: 
ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously 
like an uncondensed distance matrix
