import numpy as np
from apply_filter import *
import cv2
import matplotlib.pyplot as plt

def gaussian_filter(n,sigma2):
    x, y = np.meshgrid(np.arange(-(n//2),n//2+1),np.arange(-(n//2),n//2+1))
    filter = np.exp(-(x**2 + y**2)/(2*sigma2))
    filter *= np.ceil(1/filter[0,0])
    filter = filter.astype(int)
#     print(filter)
    return filter/np.sum(filter)

img = cv2.imread('images/zebra.jpeg',cv2.IMREAD_GRAYSCALE)

plt.figure("gaussian")
plt.subplot(121)
plt.imshow(img,cmap='gray')

filter = gaussian_filter(5,2)
r_img = apply_filter(img, filter)
plt.subplot(122)
plt.imshow(r_img, cmap='gray')
plt.show()