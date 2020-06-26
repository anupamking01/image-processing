import cv2

path1 = 'images/image_a.png'
path2 = 'images/image_b.png'

image_a = cv2.imread(path1)
image_b = cv2.imread(path2)

cv2.imshow('image_a', image_a)
cv2.waitKey()
cv2.imshow('image_b', image_b)
cv2.waitKey()

res = cv2.bitwise_or(image_a, image_b)
cv2.imshow('res', res)
cv2.waitKey()
