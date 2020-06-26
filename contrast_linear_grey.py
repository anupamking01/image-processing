import cv2
import numpy as np

img=cv2.imread('landscape.jpg',0)
int_min=np.amin(img)
int_max=np.amax(img)
x1=int_min
y1=0
x2=int_max
y2=255
m=float(y2-y1)/float(x2-x1)
out=img*m
out=out-(x1*m)

out=np.array(out, dtype = np.uint8)
print(img)
print(out)
cv2.imshow('image', img)
cv2.imshow('contrast expanded', out)
cv2.waitKey(0)
cv2.destroyAllWindows()