import cv2
import numpy as np

img = cv2.imread('Capture.png', 1)
print(img)
print(img.shape)

out = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]))
out2 = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]))
shp = img.shape

for i in range(shp[0]):
    out[2*i,:shp[1]]=np.copy(img[i,:])
    
for j in range(shp[1]):
    out2[:, 2*j] = np.copy(out[:,j])
    
out = np.copy(out2)

for i in range(img.shape[0] - 1):
    for k in range(img.shape[2]):
        out[2*i+1,:,k]=out[2*i,:,k]/2+out[2*i+2,:,k]/2

for j in range(img.shape[1]-1):
	for k in range(img.shape[2]):
		out[:,2*j+1,k]=out[:,2*j,k]/2+out[:,2*j+2,k]/2

out[-1,:]=out[-2,:]
out[:,-1]=out[:,-2]

out=np.array(out, dtype = np.uint8)

cv2.imshow('image', img)
cv2.imshow('zoomed', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
