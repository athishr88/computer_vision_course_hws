import math
import numpy as np
import cv2
import random
import scipy.spatial.distance as dist
import copy
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import struct

cluster_method = int(input("Select a cluster function\n1. K Means\n2. HAC\n:"))


def grabCat(img, segments, segment_list, background = None):
    height = segments[0].img.shape[0]
    width = segments[0].img.shape[1]

    mask = np.zeros([height,width])
    if background is None:
        background = np.zeros(img.shape)
    else:
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)

    for element in segment_list:
        mask[segments[element].mask == 1] = 1

    mask = np.uint8(mask)
    print(mask.shape)
    print(background.shape)
    cropped_image = cv2.bitwise_and(img, img, mask=mask)
    mask1 = 255*mask
    mask1 = np.invert(mask1)
    mask1[mask1 == 255] = 1
    background_cropped = cv2.bitwise_and(background, background, mask=mask1)

    final_image = cropped_image+background_cropped

    final_image = np.uint8(final_image)
    plt.figure(4)
    plt.imshow(final_image)
    plt.show()
    cv2.imshow('frame', final_image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
# def chooseSegments(segments, background=None):
#     height = segments[0].img.shape[0]
#     width = segments[0].img.shape[1]
#     mask = np.zeros([height, width])
#
#     simg = np.zeros([height, width, 3])
#
#     if background is None:
#         background = np.zeros(simg.shape)
#     else:
#         background = cv2.resize(background, [height, width], interpolation=cv2.INTER_LINEAR)
#
#     simg = copy.deepcopy(background)
#
#     h = plt.figure(3)
#     i = 0
#
#     def KeyPressFn(h, e, mask, i=0):
#         if e.key == 'h':
#             i = min(i+1, len(segments)-1)
#         elif e.key == 'g':
#             i = max(i-1, 0)
#         elif e.key == 't':
#             mask[segments[i].mask == 1] = 1
#
#     #####################################
#     plt.setp(h, 'KeypressFcn')


def makeMeanColorImage(segments):
    meanColorImg = np.zeros(segments[0].img.shape)

    for channel in range(meanColorImg.shape[2]):
        simgChannel = np.zeros([meanColorImg.shape[0], meanColorImg.shape[1]])
        for c in range(len(segments)):
            segmentChannel = np.float32(segments[c].img[:, :, channel])
            mask = segments[c].mask
            simgChannel[mask == 1] = np.mean(segmentChannel[mask == 1])
        meanColorImg[:, :, channel] = simgChannel
    meanColorImg = np.uint8(meanColorImg)

    return meanColorImg


def showMeanColorImage(img, segments):
    meanColorImage = makeMeanColorImage(segments)
    plt.figure(2)
    plt.subplot(1,2,1)
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(meanColorImage)

def makeSegments(img, idx):
    idx = np.transpose(np.uint8(idx))
    idx = idx+1
    labels = np.unique(idx)

    class Segments:
        def __init__(self, mask, img):
            self.mask = mask
            self.img = img
    segments = []
    for i in range(1, len(labels)+1):
        mask = copy.deepcopy(idx)
        mask[mask != i] = 0
        mask[mask == i] = 1
        img3 = cv2.bitwise_and(img, img, mask=mask)
        # segments[i]['img'] = np.array([i])
        segments_item = Segments(mask, img3)
        segments.append(segments_item)

    return segments


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

def KMeansCLustering(X, k, visualize=False, centers=np.array([])):
    X = np.float32(X)
    m = X.shape[0]
    n = X.shape[1]

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


def HAClustering(X, k, visualize=False):
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

        # temp = np.array([dists[i], dists[j]])
        # dists[i] = np.min(temp, axis=0)
        # dists = np.delete(dists, i, axis=0)
        # dists = np.delete(dists, i, axis=1)
        temp = np.array([dists[i], dists[j]])
        dists[i, :] = np.mean(temp, axis=0)
        dists[:, i] = np.mean(temp, axis=0)
        dists[j, :] = float('inf')
        dists[:, j] = float('inf')

        centroids[i] = (centroids[i] + centroids[j]) / 2
        cluster_sizes[i] += cluster_sizes[j]
        cluster_sizes[j] = 0
        centroids[j] = float('inf')
        idx[idx == idx[j]] = idx[i]

        num_clusters -= 1

    # Reindexing clusters
    u = np.unique(idx)
    for i in range(len(u)):
        idx[idx == u[i]] = i

    if n == 2 and visualize == True:
        visualizeClusters2D(X, idx, centroids)

    return idx


def computeColorFeatures(img):
    features = np.float32(img)
    return features


def ComputePositionColorFeatures(img):
    height = img.shape[0]
    width = img.shape[1]
    features = np.zeros([height, width, 5])
    for i in range(3):
        features[:, :, i] = img[:, :, i]


    for row in range(height):
        features[row, :, 3] = row
    for col in range(width):
        features[:, col, 4] = col

    return features


def NormalizeFeatures(features):
    features = np.float32(features)
    featureNorm = features

    for i in range(features.shape[2]):
        mean = np.mean(features[:, :, i])
        std = np.std(features[:, :, i])
        featureNorm[:, :, i] = (featureNorm[:, :, i] - mean)/std

    return featureNorm


def ComputeFeatures(img2):
    img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def gradient_x(img1):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img1 = np.float32(img1)
        grad_img = ndimage.convolve(img1, kernel)
        return grad_img


    def gradient_y(img1):
        kernel = np.transpose(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img1 = np.float32(img1)
        grad_img = ndimage.convolve(img1, kernel)
        return grad_img


    def gradient_mag(gx, gy):
        grad_img = np.hypot(gx, gy)
        return grad_img

    grad_x = gradient_x(img)
    grad_y = gradient_y(img)

    grad = gradient_mag(grad_x, grad_y)
    #
    # mean = np.mean(grad)
    # std = np.std(grad)
    # edge_features = (grad - mean)/std
    pc_features = ComputePositionColorFeatures(img2)

    features = np.dstack([pc_features, grad])
    return features
'''
    # Display gradient image
    grad = abs(255 * grad / np.max(grad))
    grad = np.uint8(grad)
    cv2.imshow('frame', grad)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
'''


def reshape(features):
    points = np.reshape(features, [features.shape[0] * features.shape[1], features.shape[2]])

    return points

def ComputeSegmentation(img, k, clustering_method, feature_fn, resize=1.0, normalize_features=False):
    height = img.shape[0]
    width = img.shape[1]

    if resize != 1.0:
        d_height = int(height * resize)
        d_width = int(width * resize)
        img_small = cv2.resize(img, (d_height, d_width), interpolation=cv2.INTER_LINEAR)
    else:
        img_small = img

    # Compute features for small image
    features = feature_fn(img_small)
    if normalize_features == True:
        features = NormalizeFeatures(features)

    # Feature is already normalized
    # Reshaping
    points = np.reshape(features, [features.shape[0]*features.shape[1], features.shape[2]])

    if clustering_method == 1:
        idx = KMeansCLustering(points, k)
    else:
        idx = HAClustering(points, k)

    # Reshaping
    idx = np.reshape(idx, [features.shape[0], features.shape[1]])
    idx = np.transpose(idx)

    # Re-scaling
    idx = cv2.resize(idx, (height, width), interpolation=cv2.INTER_LINEAR)
    segments = makeSegments(img, idx)

    return segments


def showSegments(img, segments):
    grid_width = math.ceil(math.sqrt(len(segments)+1))
    grid_height = math.ceil((1+len(segments))/grid_width)

    plt.figure(1)

    plt.subplot(grid_height, grid_width, 1)
    plt.title('Original Image')
    plt.imshow(img)

    # Show each segment
    for i in range(len(segments)):
        plt.subplot(grid_height, grid_width, i+2)
        plt.title(f'Segment {i+1}')
        plt.imshow(segments[i].img)

    plt.show()


def runComputeSegmentation():
    image = cv2.imread('studentFiles/imgs/black_kitten_star.jpg')
    k = 3
    cluster_method = 1
    feature_function = ComputeFeatures

    normalize_features = True

    resize = 1.0

    segments = ComputeSegmentation(image, k, cluster_method, feature_function, resize, normalize_features)
    showMeanColorImage(image, segments)
    showSegments(image, segments)
    return segments, image


segments, image = runComputeSegmentation()
key = int(input("Enter the segments to transfer"))
segment_list = []
while key != 0:
    segment_list.append(key)
    key = int(input("Enter the segments to transfer"))

background_image = cv2.imread('studentFiles/imgs/backgrounds/beach.jpg')
segment_list = np.array(segment_list, dtype='uint8')
segment_list = segment_list - 1
grabCat(image, segments, segment_list, background_image)