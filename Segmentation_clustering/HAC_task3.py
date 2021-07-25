import numpy as np
import scipy.io as io
import random
import scipy.spatial.distance as dist
import copy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def reIndexClusters(idx):
    u = np.unique(idx)
    for i in range(len(u)):
        idx[idx == u[i]] = i

    return idx


def visualizeClusters2D(points, idx, centers, fig_handle=False):
    if fig_handle==False:
        # fig_handle = plt.figure()
        fig_handle = 1

    POINT_SIZE = 8
    CENTER_SIZE = 30
    CONVEX_HULL_ALPHA = 0.25

    assert(points.shape[1] == 2)

    old_idx = idx
    idx = reIndexClusters(idx)

    num_clusters = len(np.unique(idx))

    R = np.linspace(0, 1, 6)
    temp = plt.cm.hsv(R)
    colors = temp[:, 0:3]

    # plt.figure(fig_handle)
    # plt.clf() # TODO doubtful

    for i in range(num_clusters):
        x = points[idx == i, 0]
        y = points[idx == i, 1]
        hull_points = points[idx == i]

        cluster_size = x.shape[0]

        plt.plot(x, y, marker='o', color=colors[i, :], markersize=POINT_SIZE)

        if cluster_size > 1 and centers.size != 0:
            j = np.where(idx == i)[0]
            j = np.int32(old_idx[j])

            plt.plot(centers[j, 0], centers[j, 1], '.', color=colors[i, :], markersize=CENTER_SIZE)

        if cluster_size == 2:
            plt.plot(x, y, color=colors[i, :])
        elif cluster_size > 2:
            hull = ConvexHull(hull_points)
            for simplex in hull.simplices:
                plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], color=colors[i, :], markersize=CONVEX_HULL_ALPHA)
    plt.show()

def HAClusteringCV(X, k, visualize):
    method = int(input("Select method:\n1. Single Link\n2. Complete Link\n3. Average Link\n:"))

    X = np.float32(X)
    m, n = X.shape
    plt.figure(1)

    num_clusters = m
    idx = np.arange(m)

    centroids = copy.deepcopy(X)
    cluster_sizes = np.ones(m)

    dists = dist.squareform(dist.pdist(centroids))
    np.fill_diagonal(dists, float('inf'))

    iteration = 0

    while num_clusters > k:
        iteration += 1
        print(iteration)
        min_dist = np.min(dists)

        i = np.where(dists == min_dist)[0][0]
        j = np.where(dists == min_dist)[0][1]

        # Make sure that i < j
        if i > j:
            t = i
            i = j
            j = t
        else:
            pass

        temp = np.array([dists[i], dists[j]])
        if method == 3:
            dists[i, :] = np.mean(temp, axis=0)
            dists[:, i] = np.mean(temp, axis=0)
            dists[j, :] = float('inf')
            dists[:, j] = float('inf')

        elif method == 1:
            dists[i, :] = np.min(temp, axis=0)
            dists[:, i] = np.min(temp, axis=0)
            dists[j, :] = float('inf')
            dists[:, j] = float('inf')
        elif method == 2:
            dists[i, :] = np.max(temp, axis=0)
            dists[:, i] = np.max(temp, axis=0)
            dists[j, :] = float('inf')
            dists[:, j] = float('inf')

        centroids[i] = (centroids[i] + centroids[j]) / 2
        cluster_sizes[i] += cluster_sizes[j]
        cluster_sizes[j] = 0
        centroids[j] = float('inf')
        idx[idx == idx[j]] = idx[i]
        np.fill_diagonal(dists, float('inf'))

        num_clusters -= 1

    # Reindexing clusters
    u = np.unique(idx)
    for i in range(len(u)):
        idx[idx == u[i]] = i

    if n == 2 and visualize == True:
        visualizeClusters2D(X, idx, centroids)

    return idx


test_data = io.loadmat('studentFiles/test_data/KMeansClusteringTest.mat')
num_clusters = 5
X = test_data['X']
X = np.array(X, dtype='float32')
num, dim = X.shape

# Setting variables
cl_centers = np.array(random.choices(X, k=num_clusters))

idx = HAClusteringCV(X, num_clusters, visualize=True)