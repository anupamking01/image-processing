import numpy as np
import cv2
from apply_filter import *
import matplotlib.pyplot as plt

precuitt_x = np.array([[-1,0,1],
                              [-1,0,1],
                              [-1,0,1]])
precuitt_y = np.array([[-1,-1,-1],
                              [0,0,0],
                              [1,1,1]])

# img = np.array([[34,66,65],
#                 [14,35,64],
#                 [12,15,42]])
img = cv2.imread('images/cups.jpg',cv2.IMREAD_GRAYSCALE)
h_pass_k = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])/9
img = apply_filter(img, h_pass_k)

plt.subplot(121)
plt.imshow(img,cmap='gray')

gx = apply_filter(img,precuitt_x,padding=True,allow_neg=True)
gy = apply_filter(img,precuitt_y,padding=True,allow_neg=True)
mag = np.abs(gx) + np.abs(gy)
dir = 180*np.arctan(gy/gx)/np.pi

# print(img)
print(gx)
# print(img)
print(gy)
print(mag)
print(dir)
plt.subplot(122)
plt.imshow(dir,cmap='Purples')
plt.colorbar()

plt.show()