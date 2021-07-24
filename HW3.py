import cv2
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

# Prompts function
def main_menu():
    # Displays these menus when prompted
    print('\n[1] Corner detection in original image')
    print('[2] Corner detection under image transformations')
    print('[3] Corner detection in filtered images')
    print('[4] Test robustness of the code')
    print('[0] Exit the program')

# Prompts function
def transform_menu():
    # Displays these menus when prompted
    print('\n[0] Go back')
    print('[1] Translate')
    print('[2] Rotate')
    print('[3] Scale')
    print('[4] Detect corner')
    print('[5] Reset')

# Prompts function
def filters_menu():
    # Displays these menus when prompted
    print('\n[0] Go back')
    print('[1] Brighten')
    print('[2] Darken')
    print('[3] Sharpen([3X3] kernel')
    print('[4] Sharpen([5X5] kernel')

# Prompts function
def noise_menu():
    # Displays these menus when prompted
    print('\n [0] Go back')
    print('[1] Small gaussian noise')
    print('[2] Medium gaussian noise')
    print('[3] High gaussian noise')
    print('[4] No noise')
    print('[5] Plot all values graph(Run 1, 2 and 3 before selecting this!)')

# Detects corner and displays
def corner_detect(image2, threshold, disp=True):
    image = copy.deepcopy(image2)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    # detects corners
    dst1 = cv2.cornerHarris(gray_img, 2, 3, .04)
    # Dilate corner pixels in a 3x3 kernel
    dst = cv2.dilate(dst1, None)
    # Marks the corners in the original image
    image[dst > threshold * dst.max()] = [0, 0, 255]
    if disp == False:
        return dst1
    else:
        # Display the image
        cv2.imshow('Corners', image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

# Returns the coordinates of corners in an image
def corner_coordinates(image2, threshold):
    image = copy.deepcopy(image2)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img, 2, 3, .04)
    # Initiate a matrix to mark the corners
    corners = np.zeros(dst.shape)
    # Filters out corners under a threshold
    corners[dst > threshold * dst.max()] = 255
    # Marks the coordinates of corners in a list and returns the result
    a = []
    a.append(np.argwhere(corners == 255))
    return a[0]

# Load the image
img = cv2.imread('chessboard1.jpg')
rows, cols, depth = img.shape

# Prompt user input of the operation they want to perform
main_menu()
option = int(input('Enter your option: '))

while option != 0:
    if option == 1:
        # Corner detection in the original image
        print("Corner detection in original image")
        corner_detect(img, 0.08)

    elif option == 2:
        # Corner detection program when images are transformed
        print('Image transformations')
        transform_menu()
        option2 = int(input('Enter the option: '))
        trfm_img = copy.deepcopy(img)
        while option2 != 0:
            # Options provided to perform translation rotation and scaling of the original image

            if option2 == 1:
                trfm_rows = trfm_img.shape[0]
                trfm_cols = trfm_img.shape[1]
                # translation based on user input
                tr_amount = int(input('Enter the percentage pixels to be translated (0 -100): '))
                tr_rows = int(trfm_rows + tr_amount * trfm_rows / 100)
                tr_cols = int(trfm_cols + tr_amount * trfm_cols / 100)
                tr_img = np.zeros([tr_rows, tr_cols, depth])

                tr_x = int(tr_amount * trfm_rows/100)
                tr_y = int(tr_amount * trfm_cols/100)

                tr_img[tr_x:tr_rows, tr_y:tr_cols, :] = trfm_img[:, :, :]
                trfm_img = np.uint8(tr_img)

                # Displays the translated image
                cv2.imshow('Transformed Image', trfm_img)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()
            elif option2 == 2:
                # Rotation based on user input
                trfm_rows = trfm_img.shape[0]
                trfm_cols = trfm_img.shape[1]
                angle = int(input('Enter the angle of rotation[0 - 360: '))
                scale = abs(trfm_rows / (trfm_cols * math.sin(math.radians(angle)) + trfm_rows * math.cos(math.radians(angle))))
                print(scale)
                T = cv2.getRotationMatrix2D((trfm_cols/2, trfm_rows/2), angle, scale)
                trfm_img = cv2.warpAffine(trfm_img, T, (trfm_cols, trfm_rows))

                # Displays the rotated image
                cv2.imshow('Transformed Image', trfm_img)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

            elif option2 == 3:
                # Scaling based on user input
                scale = float(input('Enter the scaling factor: '))
                trfm_rows = trfm_img.shape[0]
                trfm_cols = trfm_img.shape[1]
                trfm_img = cv2.resize(trfm_img, (int(scale*trfm_cols), int(scale*trfm_rows)), interpolation=cv2.INTER_LINEAR)
                # trfm_img = cv2.warpAffine(trfm_img, T, (trfm_cols, trfm_rows))

                # Displays the scaled image
                cv2.imshow('Transformed image', trfm_img)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

            elif option2 == 4:
                # Option to detect corners in the transformed image
                corner_detect(trfm_img, 0.08)

            elif option2 == 5:
                # Option to revoke all the transformations applied
                trfm_img = img

            else:
                print("\nInvalid input\n")


            transform_menu()
            option2 = int(input('Enter the option: '))

    elif option == 3:
        # Program to detect corners for image under different filters
        # Detects corners for images that are brightened darkened and sharpened in a 3x3 and 5x5 kernel
        print('Image filters')
        filters_menu()
        option3 = int(input('Enter the option: '))
        filt_img = copy.deepcopy(img)

        while option3 != 0:

            if option3 == 1:
                # Corner detection upon brightening the image with the values provided by the user
                bright_val = int(input('Enter the brightness: '))

                T = np.zeros(filt_img.shape, dtype=np.uint16)
                T += filt_img
                # Added the brightness value provided by the user
                T += bright_val
                # Normalizing
                T[T>255] = 255
                bright_img = np.array(T, dtype=np.uint8)

                corner_detect(bright_img, 0.08)

            elif option3 == 2:
                # Corner detection upon darkening the image with the values provided by the user
                dark_val = int(input('Enter the amount to be darkened: '))

                T = np.zeros(filt_img.shape, dtype=np.int32)
                T += filt_img
                # Subtracted the brightness value provided by the user
                T -= dark_val
                # Normalizing
                T[T < 0] = 0
                dark_img = np.array(T, dtype=np.uint8)

                corner_detect(dark_img, 0.08)
            elif option3 == 3:
                # Initiating the sharpened image matrix with zeros
                r, c, d = filt_img.shape
                sharp_img = np.zeros(filt_img.shape, dtype=np.int32)

                # Iterating through the cells to do sharpening with a 3x3 kernel
                # 2X current pixel - 1/9(sum of pixels in 3x3 kernel)
                for i in range(r):
                    for j in range(c):
                        for k in range(d):
                            rmin = max(0, i-1)
                            rmax = min(r-1, i+1)
                            cmin = max(0, j-1)
                            cmax = min(c-1, j+1)
                            T = filt_img[rmin:rmax, cmin:cmax, k]

                            mean_val = T.mean()
                            sharp_img[i, j, k] += int(2* filt_img[i, j, k] - mean_val)

                # Normalizing
                sharp_img[sharp_img>255] = 255
                sharp_img[sharp_img<0] = 0
                sharp_img = np.array(sharp_img, np.uint8)
                # Corner detection
                corner_detect(sharp_img, 0.08)
            elif option3 == 4:
                # Initiating the sharpened image matrix with zeros
                r, c, d = filt_img.shape
                sharp_img = np.zeros(filt_img.shape, dtype=np.int32)

                # Iterating through the cells to do sharpening with a 5x5 kernel
                # 2X current pixel - 1/25(sum of pixels in 5x5 kernel)
                for i in range(r):
                    for j in range(c):
                        for k in range(d):
                            rmin = max(0, i-2)
                            rmax = min(r-1, i+2)
                            cmin = max(0, j-2)
                            cmax = min(c-1, j+2)
                            T = filt_img[rmin:rmax, cmin:cmax, k]

                            mean_val = T.mean()
                            sharp_img[i, j, k] += int(2* filt_img[i, j, k] - mean_val)

                # Normalizing
                sharp_img[sharp_img>255] = 255
                sharp_img[sharp_img<0] = 0
                sharp_img = np.array(sharp_img, np.uint8)
                # Corner detection
                corner_detect(sharp_img, 0.08)
            else:
                print("Invalid input")

            filters_menu()
            option3 = int(input('Enter the option: '))

    elif option == 4:
        # testing the robustness of the code
        # Loading image
        wt_image = cv2.imread('white_square.jpg')

        gauss_img = copy.deepcopy(wt_image)
        r, c, d = gauss_img.shape

        # Initiating the values for gaussian noise operation
        mean = 0
        threshold = 0.4

        # Detecting the coordinates of the original image
        original_corners = np.array(corner_coordinates(wt_image, threshold))

        # Prompt menu
        noise_menu()
        option4 = int(input('Enter the option'))

        while option4 != 0:
            if option4 == 1:
                # Adding a gaussian noise with variance of 0.4 and check for deviation
                var_s = .4
                sigma = var_s**0.5

                # Adding noise to the image
                gauss = np.random.normal(mean, sigma, (r,c,d))
                gauss = gauss.reshape(r,c,d)
                noisy_img = gauss_img + gauss
                noisy_img = np.uint8(noisy_img)

                # Detecting corners for the noisy image
                noise_corners = corner_coordinates(noisy_img, threshold)

                # Detecting the normal distance of each corner from the nearest corner in the original image and
                # rms distance is added on
                dist_sum = 0
                for noise_corner in noise_corners:
                    distances = []
                    for orig_corner in original_corners:
                        distance = np.linalg.norm(noise_corner-orig_corner)
                        distances.append(distance)
                    dist_sum += np.min(distances)
                print(dist_sum)

                # Total RMS value of all detected corners are calculated
                rms_small = dist_sum

                # Displays the detected corners
                corner_detect(noisy_img, threshold)

                # Plot the RMS against the variance
                plt.plot((0,var_s), (0, rms_small))
                plt.xlabel('Variance')
                plt.ylabel('RMS of all corners detected')
                plt.show()

            elif option4 == 2:
                # Adding a gaussian noise with variance of 0.5 and check for deviation
                var_m = .5
                sigma = var_m**0.5

                # Adding noise to the image
                gauss = np.random.normal(mean, sigma, (r,c,d))
                gauss = gauss.reshape(r,c,d)
                noisy_img = gauss_img + gauss
                noisy_img = np.uint8(noisy_img)

                # Detecting corners for the noisy image
                noise_corners = corner_coordinates(noisy_img, threshold)
                # Detecting the normal distance of each corner from the nearest corner in the original image and rms
                # distance is added on
                dist_sum = 0
                for noise_corner in noise_corners:
                    distances = []
                    for orig_corner in original_corners:
                        distance = np.linalg.norm(noise_corner - orig_corner)
                        distances.append(distance)
                    dist_sum += np.min(distances)
                print(dist_sum)

                # Total RMS value of all detected corners are calculated
                rms_medium = dist_sum

                # Displays the detected corners
                corner_detect(noisy_img, threshold)

                # Plot the RMS against the variance
                plt.plot((0, var_m), (0, rms_medium))
                plt.xlabel('Variance')
                plt.ylabel('RMS of all corners detected')
                plt.show()

            elif option4 == 3:
                # Adding a gaussian noise with variance of 0.5 and check for deviation
                var_h = .6
                sigma = var_h**0.5

                # Adding noise to the image
                gauss = np.int32(np.random.normal(mean, sigma, (r,c,d)))
                noisy_img = gauss_img + gauss
                noisy_img = np.uint8(noisy_img)

                # Detecting corners for the noisy image
                noise_corners = corner_coordinates(noisy_img, threshold)
                # Detecting the normal distance of each corner from the nearest corner in the original image and rms
                # distance is added on
                dist_sum = 0
                for noise_corner in noise_corners:
                    distances = []
                    for orig_corner in original_corners:
                        distance = np.linalg.norm(noise_corner - orig_corner)
                        distances.append(distance)
                    dist_sum += np.min(distances)
                print(dist_sum)

                # Total RMS value of all detected corners are calculated
                rms_high = dist_sum

                # Displays the detected corners
                corner_detect(noisy_img, threshold)

                # Plot the RMS against the variance
                plt.plot((0, var_h), (0, rms_high))
                plt.xlabel('Variance')
                plt.ylabel('RMS of all corners detected')
                plt.show()

            elif option4 == 4:
                # Displays the corners in the original image upon prompt
                corner_detect(wt_image, threshold)

            elif option4 == 5:
                # Plots the RMS against the variance for all the noise variations
                plt.plot([var_s, var_m, var_h], [rms_small, rms_medium, rms_high])
                plt.xlabel('Variance')
                plt.ylabel('RMS of all corners detected')
                plt.show()

            else:
                print('Invalid input')
            noise_menu()
            option4 = int(input('Enter the option'))
    else:
        print('\ninvalid Option\n')
    main_menu()
    option = int(input('Enter your option: '))

print('Thank you for using the program')