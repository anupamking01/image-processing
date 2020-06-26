import numpy as np
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

def apply_filter(img,kernel,padding=True,allow_neg=False):
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