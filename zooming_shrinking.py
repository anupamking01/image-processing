import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def calc_sizes(img, ratio_x, ratio_y):
    size = img.shape
    n_size = ((img.shape[0] - 1) * ratio_y + 1, (img.shape[1] - 1) * ratio_x + 1)
    return size, n_size


def nearest_zoom(img, ratio_x, ratio_y):
    size, n_size = img.shape, (int(img.shape[0]*ratio_y), int(img.shape[1]*ratio_x))
    n_img = np.zeros(n_size)

    for i in range(n_size[0]):
        for j in range(n_size[1]):
            n_img[i][j] = img[int(i / ratio_y)][int(j / ratio_x)]
    return n_img


def bilinear_zoom(img, ratio_x, ratio_y):
    size, n_size = calc_sizes(img, ratio_x, ratio_y)
    b_img = np.zeros(n_size)
    temp = np.zeros((size[0], n_size[1]))

    for i in range(size[0]):
        for j in range(n_size[1]):
            temp[i][j] = (int(img[i][math.floor(j / ratio_x)]) + int(img[i][math.ceil(j / ratio_x)])) / 2

    for i in range(n_size[0]):
        for j in range(n_size[1]):
            b_img[i][j] = (int(temp[math.floor(i / ratio_y)][j]) + int(temp[math.ceil(i / ratio_y)][j])) / 2
    return b_img


def main():
    path = "images/rose.jpeg"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = np.array([[15,30,15],[30,15,30]])
    # img = np.arange(100).reshape((10,10))
    print(img)
    # plt.figure(1,figsize=(5,5))
    # plt.subplot(321)
    plt.figure("original")
    plt.title("original")
    # plt.grid(True, color="gray")
    plt.imshow(img, cmap="gray")
    #
    ratio_y, ratio_x = 3,3
    #
    n_img = nearest_zoom(img, ratio_x, ratio_y)
    print(n_img)
    # plt.subplot(322)
    plt.figure("nearest neighbour")
    plt.title("nearest neighbour")
    # plt.grid(True, color="gray", )
    plt.imshow(n_img, cmap="gray")
    #
    b_img = bilinear_zoom(img, ratio_x, ratio_y)
    print(b_img)
    # plt.subplot(323)
    plt.figure("bilinear interpolation")
    plt.title("bilinear interpolation")
    plt.grid(True, color="gray")
    plt.imshow(b_img, cmap="gray")

    r_img = nearest_zoom(img,1/2,1/2)
    print(r_img)
    # plt.subplot(325)
    plt.figure("reducing")
    plt.title("reducing")
    # plt.grid(True, color="gray")
    plt.imshow(r_img, cmap="gray")

    plt.show()


if __name__ == "__main__":
    main()
