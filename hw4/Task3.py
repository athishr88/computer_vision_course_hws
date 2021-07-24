import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Smoothing
'''
This code finds the disparity between two rectified images with a level of smoothing
Function works in 3 stages

1. Find the edges in the two images and find the disparity between the edges based on edge and similarity constraint
2. Find the difference between disparities in two adjacent edges and restrict the disparities between those edges within
   the calculated difference. Assign disparities of the pixels between edges with similarity constraint
3. Assign unassigned pixels a disparity value of previous pixel.
'''

# Importing images
img_left = cv2.imread('frameLeftgray.png', 0)
img_right = cv2.imread('frameRightgray.png', 0)

# Find the edges of both images
edge_left = cv2.Canny(img_left, 30, 50)
edge_right = cv2.Canny(img_right, 30, 50)

w_size = int(input('Enter the window size(Enter odd numbers): '))
disp_range = 64

# Pre-allocating disparity array with -1
disparity_array_edges = np.zeros(img_left.shape) - 1

# Gaussian blur
if (w_size > 1):
    img_left = cv2.GaussianBlur(img_left, (w_size, w_size), 0)
    img_right = cv2.GaussianBlur(img_right, (w_size, w_size), 0)

# Converting data type for numerical operations
img_left = np.int32(img_left)
img_right = np.int32(img_right)
count = 0
###############################################################################################
# Stage 1
# Traversing through the left image pixels
for row in range(img_left.shape[0] - (w_size-1)):
    print(count)
    count += 1
    for col in range(img_left.shape[1]-(w_size-1)):
        # The code will traverse till a maximum disparity of 'disp_range'

        # Finding minimum disparity among all edge pixels
        if col - disp_range >= 0:
            if edge_left[row, col] > 0:
                left_window = img_left[row:row+w_size, col:col+w_size]
                intensity_diff = []
                for disp in range(disp_range):
                    right_window = img_right[row:row+w_size, col-disp:col-disp+w_size]
                    SSD_mat = (right_window - left_window)**2
                    SSD = np.sum(SSD_mat)
                    intensity_diff.append(SSD)
                if len(intensity_diff) > 0:
                    min_diff = min(intensity_diff)
                else:
                    continue

                # Selecting the pixels with only small error
                if min_diff < 200*w_size**2:
                    if intensity_diff.count(min_diff) < 2:
                        disparity = intensity_diff.index(min_diff)
                        disparity_array_edges[row, col] = disparity
                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            pass


###############################################################################################
# Stage 2
count = 0
for row in range(img_left.shape[0]-(w_size-1)):
    print(count)
    count += 1
    # Initializing pairs list to record the disparities in adjacent edge pixels
    '''
    From col 1 to first edge    :Disparity smoothed with the disparity of first edge +/- 5
    Between two edges           :Disparity smoothed with the difference in disparity of the two edges +/- 5
    From edge to last col       :Disparity smoothed with disparity of last edge +/- 5        
    '''
    pairs = []
    for col in range(img_left.shape[1]-(w_size-1)):
        if col - disp_range >= 0:
            if disparity_array_edges[row, col] >= 0:
                if len(pairs) == 0:
                    # Smoothing from first column to first edge
                    pairs.append(int(disparity_array_edges[row, col]))
                    pairs.append(int(disparity_array_edges[row, col]))

                    for col2 in range(disp_range+5, col):
                        left_window = img_left[row:row+w_size, col2:col2+w_size]
                        intensity_diff = []

                        start = int(pairs[0])
                        for disp2 in range(pairs[0]-5, pairs[1]+5):
                            right_window = img_right[row:row+w_size, col2-disp2:col2-disp2+w_size]
                            SSD_mat = (right_window - left_window) ** 2
                            SSD = np.sum(SSD_mat)
                            intensity_diff.append(SSD)
                        if len(intensity_diff) > 0:
                            min_diff = min(intensity_diff)
                        else:
                            continue
                        if min_diff < 200 * w_size ** 2:
                            if intensity_diff.count(min_diff) < 2:
                                disparity = intensity_diff.index(min_diff) + start-5
                                disparity_array_edges[row, col2] = disparity
                            else:
                                pass
                        else:
                            pass
                    pairs = [(disparity_array_edges[row, col], col)]
                    continue
                elif len(pairs) == 1:
                    # Smoothing between two edges
                    pairs.append((disparity_array_edges[row, col], col))

                    for col2 in range(int(pairs[0][1])+1, int(pairs[1][1])):
                        left_window = img_left[row:row+w_size, col2:col2+w_size]
                        intensity_diff = []

                        disparity_pairs = [int(pairs[0][0]), int(pairs[1][0])]
                        min_disparity = min(disparity_pairs)
                        max_disparity = max(disparity_pairs)
                        for disp2 in range(min_disparity-5, max_disparity+5):
                            right_window = img_right[row:row + w_size, col2 - disp2:col2 - disp2 + w_size]
                            SSD_mat = (right_window - left_window) ** 2
                            SSD = np.sum(SSD_mat)
                            intensity_diff.append(SSD)
                        if len(intensity_diff) > 0:
                            min_diff = min(intensity_diff)
                        else:
                            continue
                        if min_diff < 200 * w_size ** 2:
                            if intensity_diff.count(min_diff) < 2:
                                disparity = intensity_diff.index(min_diff) + min_disparity-5
                                disparity_array_edges[row, col2] = disparity
                            else:
                                pass
                        else:
                            pass
                    pairs = [(disparity_array_edges[row, col], col)]
                    continue
                    # TODO
                else:
                    pass
            # Smoothing from last edge to last column
            elif col == img_left.shape[1]-(w_size-1) - 6 and len(pairs) == 1:
                for col2 in range(int(pairs[0][1]) + 1, col):
                    left_window = img_left[row:row + w_size, col2:col2 + w_size]
                    intensity_diff = []

                    disparity_pairs = [int(pairs[0][0]), int(pairs[0][0])]
                    min_disparity = min(disparity_pairs)
                    max_disparity = max(disparity_pairs)
                    for disp2 in range(min_disparity - 5, max_disparity + 5):
                        right_window = img_right[row:row + w_size, col2 - disp2:col2 - disp2 + w_size]
                        SSD_mat = (right_window - left_window) ** 2
                        SSD = np.sum(SSD_mat)
                        intensity_diff.append(SSD)
                    if len(intensity_diff) > 0:
                        min_diff = min(intensity_diff)
                    else:
                        continue
                    if min_diff < 200 * w_size ** 2:
                        if intensity_diff.count(min_diff) < 2:
                            disparity = intensity_diff.index(min_diff) + min_disparity - 5
                            disparity_array_edges[row, col2] = disparity
                        else:
                            pass
                    else:
                        pass
                continue
            else:
                pass
        else:
            pass


# Assigning the unassigned pixels a disparity of the previous column pixel
count = 0
for row in range(1, img_left.shape[0]-(w_size-1)):
    print(count)
    count += 1
    for col in range(1, img_left.shape[1]-(w_size-1)):
        if disparity_array_edges[row, col] == -1:
            disparity_array_edges[row, col] = disparity_array_edges[row, col-1]


# Normalizing
disparity_array_edges+=1
disparity_array_edges = 255 * disparity_array_edges / np.max(disparity_array_edges)
disparity_array_edges = np.uint8(disparity_array_edges)

# Display
cv2.imshow('frame', disparity_array_edges)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

c=plt.imshow(disparity_array_edges,cmap='jet')
plt.colorbar(c)
plt.show()