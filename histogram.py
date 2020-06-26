import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib qt

img = cv2.imread('images/aerial.jpg',cv2.IMREAD_GRAYSCALE)/2**(8-3)
plt.figure("original")
plt.imshow(img,cmap='gray')
plt.colorbar()
plt.show()
print(img.shape)
# des_img = cv2.imread('images/board_desired.jpg',cv2.IMREAD_GRAYSCALE)
bins_count = 8
# # plt.imshow(img,cmap='gray')
plt.figure(0)
plt.subplot(131)
plt.title("original")
freq, bins, patches = plt.hist(img.ravel(),bins=bins_count,color='red')

total_pixels = np.sum(freq)
#
# plt.subplot(122)
# plt.figure(1)
plt.subplot(132)
plt.title("desired")
dfreq = [0,0,0,0,50000,100000,195920,200000]
plt.bar(np.arange(0,8),dfreq)
# plt.show()
# dfreq, dbins, dpatches = plt.hist(des_img.ravel(),bins=bins_count)
# plt.show()
#
#
#
c1 = freq / total_pixels
prev = 0
for i in range(len(c1)):
    c1[i] = prev = prev + c1[i]
c1 *= bins[-1]
c1 = np.round(c1)

c2 = dfreq / total_pixels
prev = 0
for i in range(len(c2)):
    c2[i] = prev = prev + c2[i]
c2 *= bins[-1]
c2 = np.round(c2)

d = []
ic1, ic2 = 0, 0
while ic1 < len(c1):
    while c1[ic1] > c2[ic2]:
        ic2 += 1
    d.append(bins[ic2])
    ic1 += 1

d = np.array(d, dtype='int')

des_hist = np.zeros(bins_count)
ret_img = np.array(img, copy=True)

for i in range(len(d)):
    des_hist[d[i]] += freq[i]
    ret_img[img == i] = d[i]



plt.subplot(133)
plt.title("retrived")
plt.bar(np.arange(0,bins_count),des_hist)
plt.show()

plt.figure(1)
plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title("original")

# plt.subplot(132)
# plt.imshow(des_img,cmap='gray')
# plt.title("desired")

plt.subplot(122)
plt.imshow(ret_img,cmap='gray')
plt.title("retrived")

plt.show()