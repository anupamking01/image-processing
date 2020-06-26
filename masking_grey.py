import cv2
import numpy as np
import matplotlib.pyplot as plt

def pad(img,shp):
	p=np.zeros((shp[0]+2,shp[1]+2))
	p[1:-1,1:-1]=np.copy(img)
	p[0,1:-1],p[-1,1:-1]=img[0],img[-1]
	p[1:-1,0],p[1:-1,-1]=img[:,0],img[:,-1]
	p[0,0],p[0,-1]=img[0,0],img[0,-1]
	p[-1,0],p[-1,-1]=img[-1,0],img[-1,-1]
	return p

img=cv2.imread('cat.jpg',0)
shp=img.shape
shpm=(3,3)
mask=np.full(shpm,1)
p=pad(img,shp)
out=np.zeros((shp))

for i in range(shp[0]):
	for j in range(shp[1]):
		temp=np.multiply(p[i:i+shpm[0],j:j+shpm[1]],mask)
		temp2=temp.sum()
		out[i,j]=np.floor(temp2)
out=out/9
out=out.astype(int)
out=np.array(out, dtype = np.uint8)
print(img)
print(out)

cv2.imshow('image', img)
cv2.imshow('masked', out)
cv2.waitKey(0)
cv2.destroyAllWindows()