import cv2
import numpy

img1 = cv2.imread("images/duck1.jpg")
img2 = cv2.imread("images/duck4.jpg")

# img1 = cv2.resize(img1,(600,600), interpolation = cv2.INTER_AREA)
# img2 = cv2.resize(img2,(600,600), interpolation = cv2.INTER_AREA)

cv2.imshow('res', img1)
cv2.waitKey()
cv2.imshow('res', img2)
cv2.waitKey()

res = cv2.bitwise_xor(img1,img2)
cv2.imshow('res', res)
cv2.waitKey()
