import cv2

path = 'images/face1.jpeg'

# Load color image (BGR) and convert to gray
# img = cv2.imread(path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Load in grayscale mode
# img_gray_mode = cv2.imread(path, 0)
#
# # diff = img_gray_mode - img_gray
# diff = cv2.bitwise_and(img_gray,img_gray_mode)
#
# cv2.imshow('diff', diff)
# cv2.waitKey()

from PIL import Image

img = Image.open(path)

# img = img.convert("L")

img.show()

grey_img = Image.new("L", img.size)

for i in range(img.size[0]):
    for j in range(img.size[1]):
            r, g, b = img.getpixel((i, j))
            val = 0.2989*r + 0.5870*g + 0.1140*b
            grey_img.putpixel((i, j), round(val))

grey_img.show()

