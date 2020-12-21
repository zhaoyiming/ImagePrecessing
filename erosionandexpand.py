import cv2 as cv
import numpy as np


kernel = np.ones(shape=(5, 5))
kernel2 = np.ones(shape=(13, 13))


def dilate_bin_image(bin_image, kernel):
    # 膨胀
    kernel_size = kernel.shape[0]
    bin_image = np.array(bin_image)
    if (kernel_size % 2 == 0) or kernel_size < 1:
        raise ValueError("kernel size must be odd and bigger than 1")
    if (bin_image.max() != 255) or (bin_image.min() != 0):
        raise ValueError("input image's pixel value must be 0 or 1")
    d_image = np.zeros(shape=bin_image.shape)
    center_move = int((kernel_size - 1) / 2)
    for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
        for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
            d_image[i, j] = np.max(bin_image[i - center_move:i + center_move, j - center_move:j + center_move])
    return d_image


def erode_bin_image(bin_image, kernel):
    # 腐蚀
    kernel_size = kernel.shape[0]
    bin_image = np.array(bin_image)
    if (kernel_size % 2 == 0) or kernel_size < 1:
        raise ValueError("kernel size must be odd and bigger than 1")
    if (bin_image.max() != 255) or (bin_image.min() != 0):
        raise ValueError("input image's pixel value must be 0 or 1")
    d_image = np.zeros(shape=bin_image.shape)
    center_move = int((kernel_size - 1) / 2)
    for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
        for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
            d_image[i, j] = np.min(bin_image[i - center_move:i + center_move, j - center_move:j + center_move])
    return d_image


def thre_bin(gray_image, threshold=100):
    threshold_image = np.zeros(shape=(gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
    # loop for every pixel
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] > threshold:
                threshold_image[i][j] = 255
            else:
                threshold_image[i][j] = 0
    return threshold_image


# plt.imshow(bin_image, cmap="gray")


if __name__ == '__main__':
    new = cv.imread("1.jpg", cv.IMREAD_GRAYSCALE)

    new = 255 - new
    new = thre_bin(new)
    new = erode_bin_image(new, kernel)
    new =dilate_bin_image(new, kernel2)
    cv.imwrite("erosion2_expand.jpg", new)







