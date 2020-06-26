import numpy as np
from apply_filter import *
import cv2
import matplotlib.pyplot as plt


def apply_dilation(img, filter, padding=True):
    ksize = len(filter)
    pimg = img
    if padding:
        pimg = padd(img, ksize // 2)
    #     print(pimg)
    new_img = np.array(pimg, copy=True)

    for i in range(ksize // 2, pimg.shape[0] - ksize // 2):
        for j in range(ksize // 2, pimg.shape[1] - ksize // 2):
            #             print(i,":",j)
            if pimg[i, j] == filter[ksize // 2, ksize // 2]:
                new_img[i - ksize // 2:i + 1 + ksize // 2, j - ksize // 2:j + 1 + ksize // 2] = filter

    return new_img


def apply_erosion(img, filter):
    ksize = len(filter)
    new_img = np.array(img, copy=True)
    for i in range(ksize // 2, img.shape[0] - ksize // 2):
        for j in range(ksize // 2, img.shape[1] - ksize // 2):
            if np.array_equal(img[i - ksize // 2:i + 1 + ksize // 2, j - ksize // 2:j + 1 + ksize // 2], filter):
                new_img[i, j] = 255
    new_img[new_img != 255] = 0
    new_img[new_img == 255] = 1
    return new_img


def apply_opening(img, filter):
    erosed_img = apply_erosion(img, filter)
    dil_erosed_img = apply_dilation(erosed_img, filter)
    plt.figure("opening")

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("original")

    plt.subplot(132)
    plt.imshow(erosed_img, cmap='gray')
    plt.title("after erosion")

    plt.subplot(133)
    plt.imshow(dil_erosed_img, cmap='gray')
    plt.title("after erosion & dilation")
    plt.show()
    return dil_erosed_img


def apply_closing(img, filter):
    dilated_img = apply_dilation(img, filter)
    er_dilated_img = apply_dilation(dilated_img, filter)
    plt.figure("closing")

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("original")

    plt.subplot(132)
    plt.imshow(dilated_img, cmap='gray')
    plt.title("after dilation")

    plt.subplot(133)
    plt.imshow(er_dilated_img, cmap='gray')
    plt.title("after dilation & erosion")
    plt.show()
    return er_dilated_img

img = cv2.imread('images/shape1.bmp',cv2.IMREAD_GRAYSCALE)
img[img==255]=1

plt.imshow(img,cmap='gray')
filter = np.ones((3,3))
d_img = apply_dilation(img, filter)

plt.imshow(d_img,cmap='gray')

filter = np.ones((3,3))

apply_closing(img,filter)

apply_opening(img, filter)