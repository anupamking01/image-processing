import cv2
import numpy as np

image = cv2.imread('Capture.png', 1)

height, width, img_dim = image.shape
new_height = height // 2
new_width = width // 2

if img_dim == 3:
    new_image = np.zeros((new_height, new_width, 3), dtype = 'uint8')
elif img_dim == 2:
    new_image = np.zeros((new_height, new_width), dtype = 'uint8')
    
for row in range(new_height):
    for col in range(new_width):
        new_image[row][col] = image[row*2][col*2]

print(image.shape)
print(new_image.shape)

cv2.imshow('image', image)
cv2.imshow('shrinked', new_image)

cv2.waitKey(60000)
cv2.destroyAllWindows()