
import numpy as np
import cv2
from apply_filter import *
import matplotlib.pyplot as plt

img = cv2.imread('images/rose.jpeg',cv2.IMREAD_GRAYSCALE)
filter = np.ones((5,5))/25
# filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
r_img = apply_filter(img,filter)

plt.figure(0)
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("original")

plt.subplot(122)
plt.imshow(r_img,cmap='gray')
plt.title("blurred")

plt.show()

img = cv2.imread('images/cups.jpg',cv2.IMREAD_GRAYSCALE)
filter = np.ones((3,3))*-1
filter[1,1] = 8.1
filter /= 9
# filter = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
r_img = apply_filter(img,filter)

plt.figure(0)
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("original")

plt.subplot(122)
plt.imshow(r_img,cmap='gray')
plt.title("high pass")

plt.show()