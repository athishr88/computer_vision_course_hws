import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Calculating disparity with similarity constraint
def compute_disparity_array(img_left, img_right, w_size, disp_range):
    count = 0
    disparity_array = np.zeros(img_left.shape)
    for row in range(img_left.shape[0] - (w_size - 1)):
        print(count)
        count += 1
        for col in range(img_left.shape[1] - (w_size - 1)):

            window_left = img_left[row:row + w_size, col:col + w_size]
            disparity_list = []
            if col - disp_range >= 0:
                for dist in range(disp_range):
                    window_right = img_right[row:row + (w_size), col - dist:col - dist + (w_size)]
                    SSD_mat = (window_right - window_left) ** 2
                    SSD = np.sum(SSD_mat)
                    disparity_list.append(SSD)
                min_disparity = min(disparity_list)
                disparity = disparity_list.index(min_disparity)
                disparity_array[row, col] = disparity
            else:
                pass
    disparity_array = 255 * disparity_array / (np.max(disparity_array) + 1)
    disparity_array = np.uint8(disparity_array)
    return disparity_array


img_left = cv2.imread('im2.png', 0)
img_right = cv2.imread('im6.png', 0)

w_size = int(input('Enter the window size(Enter odd numbers): '))
disp_range = 64

if (w_size > 1):
    img_left = cv2.GaussianBlur(img_left, (w_size, w_size), 0)
    img_right = cv2.GaussianBlur(img_right, (w_size, w_size), 0)

img_left = np.int32(img_left)
img_right = np.int32(img_right)

######################################################################################
disparity_array_t1 = compute_disparity_array(img_left, img_right, w_size, disp_range)
######################################################################################

# Calculating disparity based on uniqueness constraint
img_left2 = cv2.imread('im2.png', 0)
img_right2 = cv2.imread('im6.png', 0)

disparity_array = np.zeros(img_left2.shape)
uniqueness_array = np.zeros(img_left2.shape)

if (w_size > 1):
    img_left = cv2.GaussianBlur(img_left2, (w_size, w_size), 0)
    img_right = cv2.GaussianBlur(img_right2, (w_size, w_size), 0)

img_left = np.int32(img_left2)
img_right = np.int32(img_right2)
count = 0

for row in range(img_left.shape[0] - (w_size-1)):
    print(count)
    count += 1
    for col in range(img_left.shape[1]-(w_size-1)):
        window_left = img_left[row:row+w_size, col:col+w_size]
        disparity_list = []
        if col - disp_range >= 0:
            for dist in range(disp_range):
                window_right = img_right[row:row+(w_size), col-dist:col-dist+(w_size)]
                SSD_mat = (window_right - window_left)**2
                SSD = np.sum(SSD_mat)
                disparity_list.append(SSD)
            flag = 0
            while flag == 0:
                if len(disparity_list) == 0:
                    min_disparity = 0
                min_disparity = min(disparity_list)
                if uniqueness_array[row, col-disparity_list.index(min_disparity)] == 0:
                    min_disp_index = disparity_list.index(min_disparity)
                    flag = 1
                else:
                    disparity_list.remove(min_disparity)

            if min_disparity < 100*w_size**2:
                disparity = min_disp_index
                disparity_array[row, col] = disparity
                uniqueness_array[row, col - disparity] = 1
            else:
                disparity_array[row, col] = disparity_array[row, col-1]
        else:
            pass

disparity_array = 255 * disparity_array/(np.max(disparity_array)+1)

#############################################################################
disparity_array_t2 = np.uint8(disparity_array)
#############################################################################

# Calculating disparity based on smoothing constraint
img_left3 = cv2.imread('im2.png', 0)
img_right3 = cv2.imread('im6.png', 0)

edge_left = cv2.Canny(img_left3, 30, 50)
edge_right = cv2.Canny(img_right3, 30, 50)

disparity_array_edges = np.zeros(img_left3.shape) - 1

if (w_size > 1):
    img_left3 = cv2.GaussianBlur(img_left3, (w_size, w_size), 0)
    img_right3 = cv2.GaussianBlur(img_right3, (w_size, w_size), 0)

img_left = np.int32(img_left3)
img_right = np.int32(img_right3)
count = 0

for row in range(img_left.shape[0] - (w_size-1)):
    print(count)
    count += 1
    for col in range(img_left.shape[1]-(w_size-1)):
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

count = 0
for row in range(img_left.shape[0]-(w_size-1)):
    print(count)
    count += 1
    pairs = []
    for col in range(img_left.shape[1]-(w_size-1)):
        if col - disp_range >= 0:
            if disparity_array_edges[row, col] >= 0:
                if len(pairs) == 0:
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
                    pairs.append((disparity_array_edges[row, col], col))

                    for col2 in range(int(pairs[0][1])+1, int(pairs[1][1])):
                        left_window = img_left[row:row+w_size, col2:col2+w_size]
                        intensity_diff = []

                        disparity_pairs = [int(pairs[0][0]), int(pairs[1][0])]
                        min_disparity = min(disparity_pairs)
                        max_disparity = max(disparity_pairs)
                        for disp2 in range(min_disparity-5, max_disparity+5):
                            right_window = img_right[row:row + w_size, col2 - disp2:col2 - disp2 + w_size]
                            if left_window.size == w_size*w_size and right_window.size == w_size*w_size:
                                SSD_mat = (right_window - left_window) ** 2
                            else:
                                SSD_mat = [0]
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

count = 0
for row in range(1, img_left.shape[0]-(w_size-1)):
    print(count)
    count += 1
    for col in range(1, img_left.shape[1]-(w_size-1)):
        if disparity_array_edges[row, col] == -1:
            disparity_array_edges[row, col] = disparity_array_edges[row, col-1]


disparity_array_edges+=1
disparity_array_edges = 255 * disparity_array_edges / np.max(disparity_array_edges)
################################################################################
disparity_array_t3= np.uint8(disparity_array_edges)
################################################################################
# Juxtaposing the disparity maps from all methods

gt = cv2.imread('disp2.png', 0)

plt.subplot(2, 2, 1)
plt.title('Similarity constraint')
plt.imshow(disparity_array_t1, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('Uniqueness constraint')
plt.imshow(disparity_array_t2, cmap='gray')
plt.subplot(2, 2, 3)
plt.title('Smoothing')
plt.imshow(disparity_array_t3, cmap='gray')
plt.subplot(2, 2, 4)
plt.title('Ground Truth')
plt.imshow(gt, cmap='gray')
plt.show()

################################################################################
# Displaying the errors of all three disparity methods

disparity_array_t1 = np.int32(disparity_array_t1)
disparity_array_t2 = np.int32(disparity_array_t2)
disparity_array_t3 = np.int32(disparity_array_t3)
gt = np.int32(gt)

t1_error = abs(disparity_array_t1 - gt)
t2_error = abs(disparity_array_t2 - gt)
t3_error = abs(disparity_array_t3 - gt)

t1_error = np.uint8(t1_error)
t2_error = np.uint8(t2_error)
t3_error = np.uint8(t3_error)

plt.subplot(2, 2, 1)
plt.title('Similarity constraint error')
plt.imshow(disparity_array_t1, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('Uniqueness constraint error')
plt.imshow(disparity_array_t2, cmap='gray')
plt.subplot(2, 2, 3)
plt.title('Smoothing error')
plt.imshow(disparity_array_t3, cmap='gray')
plt.show()
################################################################################

# Histogram subplots

t1_error_hist = t1_error.flatten()
t2_error_hist = t2_error.flatten()
t3_error_hist = t3_error.flatten()

plt.subplot(2, 2, 1)
plt.hist(t1_error_hist, bins = 30)
plt.title("Similarity error histogram")
plt.subplot(2, 2, 2)
plt.hist(t2_error_hist, bins = 30)
plt.title("Uniqueness error histogram")
plt.subplot(2, 2, 3)
plt.hist(t3_error_hist, bins = 30)
plt.title("Smoothing error histogram")
plt.show()
