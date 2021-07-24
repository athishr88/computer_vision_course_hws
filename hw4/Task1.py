import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to return disparity map from rectified left and right images with provided window size and disparity range.
def compute_disparity_array(img_left, img_right, w_size, disp_range):
    count = 0

    # Initializing disparity image matrix with all zeros
    disparity_array = np.zeros(img_left.shape)

    # Traversing through left image pixels
    for row in range(img_left.shape[0] - (w_size - 1)):
        print(count)
        count += 1
        for col in range(img_left.shape[1] - (w_size - 1)):
            window_left = img_left[row:row + w_size, col:col + w_size]
            disparity_list = []
            if col - disp_range >= 0:
                # The code will traverse till a maximum disparity of 'disp_range'

                # Finding minimum disparity among all traversed pixels
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
    # Normalizing
    disparity_array = 255 * disparity_array / (np.max(disparity_array) + 1)
    disparity_array = np.uint8(disparity_array)
    return disparity_array


# Import the rectified images
img_left = cv2.imread('frameLeftgray.png', 0)
img_right = cv2.imread('frameRightgray.png', 0)

# Variables
w_size = int(input('Enter the window size(Enter odd numbers): '))
disp_range = 64

# Apply gaussian blur
if (w_size > 1):
    img_left = cv2.GaussianBlur(img_left, (w_size, w_size), 0)
    img_right = cv2.GaussianBlur(img_right, (w_size, w_size), 0)

# Converting data type for numerical operations
img_left = np.int32(img_left)
img_right = np.int32(img_right)

# Get disparity array
disparity_array = compute_disparity_array(img_left, img_right, w_size, disp_range)

# Display
cv2.imshow('frame', disparity_array)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

c=plt.imshow(disparity_array,cmap='jet')
plt.colorbar(c)
plt.show()