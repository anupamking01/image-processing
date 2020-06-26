# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:20:55 2019

@author: AMIT
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
############### util #################
def padd(img,size):
    padd_img = np.ones((img.shape[0]+2*size,img.shape[1]+2*size),dtype=img.dtype)
    padd_img[size:img.shape[0]+size,size:img.shape[1]+size] = img
    for i in range(img.shape[1]):
        padd_img[0:size,size+i] = np.full(size,padd_img[size,size+i])
        padd_img[-size:,size+i] = np.full(size,padd_img[-size-1,size+i])
    for i in range(padd_img.shape[0]):
        padd_img[i,0:size] = np.full(size,padd_img[i,size])
        padd_img[i,-size:] = np.full(size,padd_img[i,-size-1])
    return padd_img

def apply_kernel(img,kernel,padding=True,allow_neg=False):
    ksize = len(kernel)
    pimg = img
    if padding:
        pimg = padd(img,ksize//2)
#     print(pimg)
    new_img = np.array(pimg,copy=True)
    
    for i in range(ksize//2,pimg.shape[0]-ksize//2):
        for j in range(ksize//2,pimg.shape[1]-ksize//2):
#             print(i,":",j)
            val = np.sum(pimg[i-ksize//2:i+1+ksize//2,j-ksize//2:j+1+ksize//2]*kernel)
            if allow_neg:
                new_img[i,j] = val
            else:
                new_img[i,j] = 0 if val<0 else val
#     print(new_img.shape,":",ksize//2)
#     print(new_img[1:-1,1:-1])
    return new_img[ksize//2:-(ksize//2),ksize//2:-(ksize//2)]

########### zoom / shrink ##############33
def calc_sizes(img, ratio_x, ratio_y):
    size = img.shape
    n_size = ((img.shape[0] - 1) * ratio_y + 1, (img.shape[1] - 1) * ratio_x + 1)
    return size, n_size


def nearest_zoom(img, ratio_x, ratio_y):
    size, n_size = img.shape, (int(img.shape[0]*ratio_y), int(img.shape[1]*ratio_x))
    n_img = np.zeros(n_size)

    for i in range(n_size[0]):
        for j in range(n_size[1]):
            n_img[i][j] = img[int(i / ratio_y)][int(j / ratio_x)]
    return n_img


def bilinear_zoom(img, ratio_x, ratio_y):
    size, n_size = calc_sizes(img, ratio_x, ratio_y)
    b_img = np.zeros(n_size)
    temp = np.zeros((size[0], n_size[1]))

    for i in range(size[0]):
        for j in range(n_size[1]):
            temp[i][j] = (int(img[i][math.floor(j / ratio_x)]) + int(img[i][math.ceil(j / ratio_x)])) / 2

    for i in range(n_size[0]):
        for j in range(n_size[1]):
            b_img[i][j] = (int(temp[math.floor(i / ratio_y)][j]) + int(temp[math.ceil(i / ratio_y)][j])) / 2
    return b_img


def k_zoom(img, ratio_x, ratio_y):
    size, n_size = calc_sizes(img, ratio_x, ratio_y)
    k_img = np.zeros(n_size)
    temp = np.zeros((size[0], n_size[1]))

    for i in range(size[0]):
        count = 0
        op = 0
        for j in range(n_size[1]):
            if count == 0:
                temp[i][j] = img[i][int(j / ratio_x)]
                if j < n_size[1] - 1:
                    op = (int(img[i][int(j / ratio_x) + 1]) - int(img[i][int(j / ratio_x)])) / ratio_x
            else:
                temp[i][j] = int(temp[i][j - 1] + op)
            count = (count + 1) % ratio_x

    for j in range(n_size[1]):
        count = 0
        op = 0
        for i in range(n_size[0]):
            if count == 0:
                k_img[i][j] = temp[int(i/ratio_y)][j]
                if i < n_size[0] -1:
                    op = (temp[int(i/ratio_y)+1][j] - temp[int(i/ratio_y)][j])/ratio_y
            else:
                k_img[i][j] = int(k_img[i-1][j] + op)
            count = (count+1)%ratio_y

    # k_img = temp
    return k_img

def shrink(img, ratio_x, ratio_y):
    return nearest_zoom(img,1/ratio_x,1/ratio_y)

################# neg
def negative(img):
    return 255 - img

def contrast_strech_gray(imgs,r1,r2,s1,s2):
    alpha = s1/r1
    beta = (s2-s1)/(r2-r1)
    gamma = (255-s2)/(255-r2)
    
    res = np.array(imgs, copy=True)
    for k,img in enumerate(imgs):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] < r1:
                    res[k][i][j] = alpha*img[i][j]
                elif img[i][j] < r2:
                    res[k][i][j] = beta*(img[i][j] - r1) + s1
                else:
                    res[k][i][j] = gamma*(img[i][j] - r2) + s2
    return res

def intensity_slice(imgs,r1,r2,reverse=False):
    res = np.array(imgs, copy=True)
    res[imgs < r1] = 255 if reverse else 0
    res[imgs > r2] = 255 if reverse else 0
    res[res != (255 if reverse else 0)] = 0 if reverse else 255
    return res

def hist_equal(img):
    bins_count = 256
    freq, bins, patches = plt.hist(img.ravel(),bins=bins_count,color='red')

    total_pixels = np.sum(freq)
    
    
    c1 = freq / total_pixels
    prev = 0
    for i in range(len(c1)):
        c1[i] = prev = prev + c1[i]
    c1 *= bins[-1]
    c1 = np.round(c1)
    
    imgn = np.array(img,copy=True)
    for i in range(bins_count):
        imgn[img == i] = c1[i]
        
    return imgn
    
def color_to_gray(img):
    r,g,b = img[:,:,2], img[:,:,1], img[:,:,0]
    return 0.2989*r + 0.5870*g + 0.1140*b

def apply_dilation(img,padding=True):
    kernel = np.ones((3,3))
    ksize = len(kernel)
    pimg = img
    if padding:
        pimg = padd(img,ksize//2)
#     print(pimg)
    new_img = np.array(pimg,copy=True)
    
    for i in range(ksize//2,pimg.shape[0]-ksize//2):
        for j in range(ksize//2,pimg.shape[1]-ksize//2):
#             print(i,":",j)
            if pimg[i,j] == kernel[ksize//2,ksize//2]:
                new_img[i-ksize//2:i+1+ksize//2,j-ksize//2:j+1+ksize//2] = kernel
            
    return new_img

def apply_erosion(img):
    kernel = np.ones((3,3))
    ksize = len(kernel)
    new_img = np.array(img,copy=True)
    for i in range(ksize//2,img.shape[0]-ksize//2):
        for j in range(ksize//2,img.shape[1]-ksize//2):
            if np.array_equal(img[i-ksize//2:i+1+ksize//2,j-ksize//2:j+1+ksize//2], kernel):
                new_img[i,j] = 255
    new_img[new_img != 255] = 0
    new_img[new_img == 255] = 1
    return new_img

def apply_opening(img):
    erosed_img = apply_erosion(img)
    dil_erosed_img = apply_dilation(erosed_img)
    return dil_erosed_img

def apply_closing(img):
    dilated_img = apply_dilation(img)
    er_dilated_img = apply_dilation(dilated_img)
    return er_dilated_img

def high_pass(img):
    kernel = np.ones((3,3))*-1
    kernel[1,1] = 8
    kernel /= 9
    return apply_kernel(img,kernel)

def low_pass(img):
    kernel = np.ones((7,7))/49
    return apply_kernel(img,kernel)

def call_corres_fn(img,args):
#    temp=[None]
#    args=temp.extend(args)
    try:
        args.extend([0,0,0,0,0])
        fn = {'1':[nearest_zoom,[img,int(args[1]),int(args[2])]],
              '2':[bilinear_zoom,[img,int(args[1]),int(args[2])]],
              '3':[k_zoom,[img,int(args[1]),int(args[2])]],
              '4':[shrink,[img,int(args[1]),int(args[2])]],
              '5':[negative,[img]],
              '6':[contrast_strech_gray,[img,int(args[1]),int(args[2]),int(args[3]),int(args[4])]],
              '7':[intensity_slice,[img,int(args[1]),int(args[2])]],
              '8':[hist_equal,[img]],
              '9':[color_to_gray,[img]],
              '10':[apply_dilation,[img]],
              '11':[apply_erosion,[img]],
              '12':[apply_opening,[img]],
              '13':[apply_closing,[img]],
              '14':[high_pass,[img]],
              '15':[low_pass,[img]]}
        
        return fn[args[0]][0](*(fn[args[0]][1]))
    except Exception as e:
        raise e
#        return img


def main():
    img = None
    res_img = None
    while True:
        choice=input(
 """-1.exit.
 0.import image.
 1.nearest zoom.
 2.bilinear zoom.
 3.k_zoom.
 4.shrink image.
 5.negative image.
 6.contrast strech.
 7.intensity slice.
 8.histogram equalization.
 9.color to gray.
 10.dilation.
 11.erosion.
 12.opening.
 13.closing.
 14.high pass.
 15.low pass.
 enter choice:""")
        if choice == '-1':
            break;
        elif choice == '0':
            file = input("enter file location:")
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        elif choice == '16':
            file = input("enter file location:")
            cv2.imwrite(file,img)
        elif choice in ['1','2','3','4','7']:
            if choice != '7':
                vals = input("enter ratio_x,ration_y:").split()
            else:
                vals = input("enter r1, r2:").split()
            img = call_corres_fn(img,[choice,vals[0],vals[1]])
        elif choice == '6':
            vals = input("enter r1 s1 r2 s2:").split()
            img = call_corres_fn(img, [choice,vals[0],vals[1],vals[2],vals[3]])
        else:
            img = call_corres_fn(img,[choice])
        
        cv2.imshow("result_image",np.array(img,dtype='uint8'))
        cv2.waitKey()
                    
#    args=sys.argv[1:]
#    print(args)
#    if(len(args)>=3):
#        img = cv2.imread(args[0],cv2.IMREAD_GRAYSCALE)
#        
#        res_img = call_corres_fn(img,args[1:-1])
#        cv2.imshow("original_image",img)
#        cv2.imshow("result_image",res_img)
#        cv2.waitKey()
#        cv2.imwrite(args[-1],res_img)
    
if __name__ == "__main__":
    main()




    
