import pylab as plt
import numpy as np
import matplotlib.image as mpimg

img = np.uint8(mpimg.imread('Capture.png')*255.0)

# Convert to greyScale
img = np.uint8((0.2126* img[:,:,0]) + \
  		np.uint8(0.7152 * img[:,:,1]) +\
			 np.uint8(0.0722 * img[:,:,2]))


# To calculate normalized histogram of image
def imhist(im):
    m, n = im.shape
    h = [0.0]*256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h)/(m*n)

# To find cummulative sum of a numpy array
def cummulative(h):
    return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cummulative(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new histogram, 
	# and transform function
	return Y , h, H, sk

new_img, h, new_h, sk = histeq(img)

# show old and new image
# show original image
# plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.set_cmap('gray')

# show original image
# plt.subplot(122)
plt.imshow(new_img)
plt.title('hist. equalized image')
plt.set_cmap('gray')
plt.show()

# plot histograms and transfer function
# original histogram
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h)
plt.title('Original histogram')

# hist of eqlauized image
fig.add_subplot(222)
plt.plot(new_h)
plt.title('New histogram')

# transfer function
fig.add_subplot(223)
plt.plot(sk)
plt.title('Transfer function') 

plt.show()
        
    
