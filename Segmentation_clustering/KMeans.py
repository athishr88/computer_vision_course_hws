import numpy as np
import scipy.spatial
import scipy.io as io
import cv2
import random
import scipy.spatial.distance as dist
import copy
from scipy import ndimage
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

def KMeansCLustering(X, k, visualize=False, centers=False):
    X = np.float32(X)
    m, n = X.shape

    if centers.size == 0:
        centers = np.array(random.choices(X, k=k))
    else:
        pass

    idx = np.zeros([m])
    plt.figure(1)

    iter = 0
    MAX_ITER = 100

    while True:
        old_idx = idx

        # Allotting clusters
        for i in range(m):
            min_dist = float('inf')
            for c in range(k):
                distance = np.linalg.norm(X[i] - centers[c])
                if distance < min_dist:
                    min_dist = distance
                    idx[i] = c


        # Updating cluster centers
        for cen in range(k):
            new_centre_members = []
            for i in range(m):
                if idx[i] == cen:
                    new_centre_members.append(X[i])
                else:
                    pass
            new_centre_members = np.array(new_centre_members)
            centers[cen] = np.mean(new_centre_members, axis=0)

        # Display
        if n == 2 and visualize==True:
            visualizeClusters2D(X, idx, centers)

        # End condition
        if np.array_equal(idx, old_idx):
            break

        # Stop early
        iter += 1
        if iter > MAX_ITER:
            break
    return idx


test_data = io.loadmat('studentFiles/test_data/KMeansClusteringTest.mat')
num_clusters = 5
X = test_data['X']
X = np.array(X, dtype='float32')
num, dim = X.shape

# Setting variables
cl_centers = np.array(random.choices(X, k=num_clusters))

idx = KMeansCLustering(X, num_clusters, visualize=True, centers=cl_centers)