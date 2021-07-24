import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Importing images
img_left = cv2.imread('frameLeftgray.png', 0)
img_right = cv2.imread('frameRightgray.png', 0)

# Setting variables
w_size = int(input('Enter the window size(Enter odd numbers): '))
disp_range = 64

# Initialize disparity array and uniqueness array
disparity_array = np.zeros(img_left.shape)
uniqueness_array = np.zeros(img_left.shape)

# Gaussian blur
if (w_size > 1):
    img_left = cv2.GaussianBlur(img_left, (w_size, w_size), 0)
    img_right = cv2.GaussianBlur(img_right, (w_size, w_size), 0)

# Converting data type for numerical operations
img_left = np.int32(img_left)
img_right = np.int32(img_right)

count = 0

# Traversing through left image pixels
for row in range(img_left.shape[0] - (w_size-1)):
    print(count)
    count += 1
    for col in range(img_left.shape[1]-(w_size-1)):
        window_left = img_left[row:row+w_size, col:col+w_size]
        disparity_list = []
        if col - disp_range >= 0:
            for dist in range(disp_range):
                # The code will traverse till a maximum disparity of 'disp_range'

                # Finding minimum disparity among all traversed pixels
                window_right = img_right[row:row+(w_size), col-dist:col-dist+(w_size)]
                SSD_mat = (window_right - window_left)**2
                SSD = np.sum(SSD_mat)
                disparity_list.append(SSD)

            # Checking uniqueness of the right image pixel selected, with uniqueness array
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

            # Selecting the pixels with only small error
            if min_disparity < 100*w_size**2:
                disparity = min_disp_index
                disparity_array[row, col] = disparity
                uniqueness_array[row, col - disparity] = 1
            else:
                disparity_array[row, col] = disparity_array[row, col-1]
        else:
            pass

# Normalizing
disparity_array = 255 * disparity_array/(np.max(disparity_array)+1)
disparity_array = np.uint8(disparity_array)

# Display
cv2.imshow('frame', disparity_array)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

uniqueness_array = 255 * uniqueness_array
uniqueness_array = np.uint8(uniqueness_array)
cv2.imshow('frame', uniqueness_array)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

c=plt.imshow(disparity_array,cmap='jet')
plt.colorbar(c)
plt.show()