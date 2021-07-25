import numpy as np
import cv2
import random
import time

start = time.time()

# Returns the best disparity maps in a pair of rows from rectified images based on similarity and ordering
def stereoDP(e1, e2, max_disp=32, occ=200):
    # 0 -north 1-east, 2 - NE
    # Initializing the disparity matrix and the direction of minimum cost
    D = np.zeros([e1.shape[0]+1, e2.shape[0]+1]) + 1000000
    direction_matrix = np.zeros(D.shape)

    # Pre-allocating the first row first column and pixel at [1,1]
    D[1, 1] = (e1[1] - e2[1]) ** 2
    for i in range(1, e1.shape[0]):
        for j in range(1, e2.shape[0]):
            if j-max_disp < i < j+max_disp and i-max_disp < j < i+max_disp:
                D[0, j] = j * occ
                D[i, 0] = i * occ

                # Cost calculation
                cost = (e1[i] - e2[j])**2
                # Calculating minimum cost and direction
                # If two equals, then the direction will be north
                D[i, j] = min(D[i-1, j-1] + cost, D[i-1, j] + occ, D[i, j-1] + occ)
                if D[i, j] == D[i-1, j-1] + cost:
                    direction_matrix[i, j] = 2
                elif D[i, j] == D[i-1, j] + occ:
                    direction_matrix[i, j] = 0
                else:
                    direction_matrix[i, j] = 1
            else:
                pass

    #Backtracking
    disparity_array = np.zeros(e1.shape[0])
    i = e1.shape[0]-5
    j = e1.shape[0]-5
    while i > 0 and j > 0:
        if direction_matrix[i, j] == 2:
            disparity_array[i] = abs(i - j)
            i = i-1
            j = j-1
        elif direction_matrix[i, j] == 1:
            j = j-1
        else:
            i = i-1
    return disparity_array


# Import the rectified images
img_left = cv2.imread('frameLeftgray.png', 0)
img_right = cv2.imread('frameRightgray.png', 0)

# Converting data type for numerical operations
img_left = np.int32(img_left)
img_right = np.int32(img_right)

# Getting disparity matrix from the stereoDP() function
disparity_matrix = np.zeros(img_left.shape)
count = 0
for row in range(img_left.shape[0]):
    print(count)
    count += 1
    disparity_array = stereoDP(img_left[row], img_right[row])
    disparity_matrix[row] = disparity_array

end = time.time()
duration = end - start
print(duration)

# Normalizing
disparity_matrix = 255 * disparity_matrix/(np.max(disparity_matrix))
disparity_matrix = np.uint8(disparity_matrix)

# Displaying the map
cv2.imshow('frame', disparity_matrix)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# Displaying occlusions in red
rows = disparity_matrix.shape[0]
cols = disparity_matrix.shape[1]
color_image = np.zeros([rows, cols, 3])

color_image[:, :, 1] = disparity_matrix
color_image[:, :, 0] = disparity_matrix

disparity_matrix[disparity_matrix == 0] = 255
color_image[:, :, 2] = disparity_matrix
color_image = np.uint8(color_image)
cv2.imshow('frame', color_image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
