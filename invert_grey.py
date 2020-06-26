import cv2
import numpy as np

def invertor(value):
	return 255-value

img=cv2.imread('cat.jpeg',0)
out=np.zeros(img.shape)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		out[i,j]=invertor(img[i,j])

out=np.array(out, dtype = np.uint8)

cv2.imshow('image', img)
cv2.imshow('inverted', out)
cv2.waitKey(0)
cv2.destroyAllWindows()


