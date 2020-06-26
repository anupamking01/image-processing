import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    image = np.asarray(Image.open('chap.jpg').convert("L"))
    h,w = image.shape
    img = np.zeros((h,w))
    new_image_1 = np.zeros((h,w))
    new_image_2 = np.zeros((h,w))
    new_image_3 = np.zeros((h,w))
    new_image_4 = np.zeros((h,w))
    new_image = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            if image[i,j] >0:
                img[i,j] = 255
            else:
                img[i,j] = 0
    print(image)
    print(img.shape)
    im = plt.imshow(img, cmap='gray')
    plt.show()

    for i in range(h):
        for j in range(w):
            if i==0 or i==h-1 or j==0 or j==w-1:
                new_image_1[i,j]=img[i,j]
            else:
                new_image_1[i,j]=(-img[i-1,j-1]-0*img[i-1,j]+img[i-1,j+1]-2*img[i,j-1]+0*img[i,j]+2*img[i,j+1]-img[i+1,j-1]-0*img[i+1,j]+img[i+1,j+1])/9
            if new_image_1[i,j]<=0:
                new_image_1[i,j]=0
            else:
                new_image_1[i,j]=255
    new_image_1=255-new_image_1
    im = plt.imshow(new_image_1, cmap='gray')
    plt.show()
    
    for i in range(h):
        for j in range(w):
            if i==0 or i==h-1 or j==0 or j==w-1:
                new_image_2[i,j]=img[i,j]
            else:
                new_image_2[i,j]=(-img[i-1,j-1]-2*img[i-1,j]-img[i-1,j+1]-0*img[i,j-1]+0*img[i,j]+0*img[i,j+1]+img[i+1,j-1]+2*img[i+1,j]+img[i+1,j+1])/9
            if new_image_2[i,j]<=0:
                new_image_2[i,j]=0
            else:
                new_image_2[i,j]=255
    new_image_2=255-new_image_2
    im = plt.imshow(new_image_2, cmap='gray')
    plt.show()
    
    for i in range(h):
        for j in range(w):
            if i==0 or i==h-1 or j==0 or j==w-1:
                new_image_3[i,j]=img[i,j]
            else:
                new_image_3[i,j]=(img[i-1,j-1]-0*img[i-1,j]-img[i-1,j+1]+2*img[i,j-1]+0*img[i,j]-2*img[i,j+1]+img[i+1,j-1]+0*img[i+1,j]-img[i+1,j+1])/9
            if new_image_3[i,j]<=0:
                new_image_3[i,j]=0
            else:
                new_image_3[i,j]=255
    new_image_3=255-new_image_3
    im = plt.imshow(new_image_3, cmap='gray')
    plt.show()
    
    for i in range(h):
        for j in range(w):
            if i==0 or i==h-1 or j==0 or j==w-1:
                new_image_4[i,j]=img[i,j]
            else:
                new_image_4[i,j]=(img[i-1,j-1]+2*img[i-1,j]+img[i-1,j+1]-0*img[i,j-1]+0*img[i,j]+0*img[i,j+1]-img[i+1,j-1]-2*img[i+1,j]-img[i+1,j+1])/9
            if new_image_4[i,j]<=0:
                new_image_4[i,j]=0
            else:
                new_image_4[i,j]=255
    new_image_4=255-new_image_4
    im = plt.imshow(new_image_4, cmap='gray')
    plt.show()
    
    for i in range(h):
        for j in range(w):
            if new_image_1[i,j]==0 or new_image_2[i,j]==0 or new_image_3[i,j]==0 or new_image_4[i,j]==0:
                new_image[i,j]=0
            else:
                new_image[i,j]=255
    im = plt.imshow(new_image, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()

