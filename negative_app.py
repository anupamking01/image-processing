import cv2
import numpy
import matplotlib.pyplot as plt

path = "images/pancreaticcancer.jpg"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
img = 255 - img
plt.figure(1)
plt.imshow(img,cmap="gray")

img = 255 - img
plt.figure(2)
plt.imshow(img,cmap="gray")

plt.show()
