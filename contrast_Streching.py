import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "images/rose.jpeg"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

# img = np.random.randint(50,200,(10,10))
plt.figure(1)
plt.title("original")
plt.imshow(img,cmap="gray")


# s1,s2 = 0, 255
# r1,r2 = np.min(img),np.max(img)
#
# img = s1 + ((s2-s1)/(r2-r1))*(img-r1)
# print(np.min(img),np.max(img))

r1, r2, = 130, 180
s1, s2 = 170, 210

alpha = s1/r1
beta = (s2-s1)/(r2-r1)
gamma = (255-s2)/(255-r2)

# img = np.array(img,dtype='int')

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] < r1:
            img[i][j] = alpha*img[i][j]
        elif img[i][j] < r2:
            img[i][j] = beta*(img[i][j] - r1) + s1
        else:
            img[i][j] = gamma*(img[i][j] - r2) + s2

plt.figure(2)
plt.title("enhanced")
plt.imshow(img,cmap="gray")

plt.show()