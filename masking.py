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


shp=(25,25)
img = np.floor(np.random.random(shp)*255)


shpm=(3,3)
mask=np.full(shpm,1)
p=pad(img,shp)
out=np.zeros((shp))

for i in range(shp[0]):
	for j in range(shp[1]):
		temp=np.multiply(p[i:i+shpm[0],j:j+shpm[1]],mask)
		temp2=temp.sum()
		out[i,j]=temp2

out=out/9
out=out.astype(int)
fig = plt.figure(100)
fig.canvas.set_window_title('Original image')
plt.imshow(img, cmap="Greys")

fig = plt.figure(200)
fig.canvas.set_window_title('Masked')
plt.imshow(out, cmap="Greys")

plt.show()

