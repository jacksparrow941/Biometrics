
import cv2
import numpy as np
from scipy import signal



def image_Quality(img,variance_thresold,d1=0.20):
    
    block_row_size=3
    block_coloumn_size=3
    #img should be gray
    m,n=img.shape
    ints_val=0
    for i in range(m):
        for j in range(n):
            ints_val+=(img[i][j])  #intensity value   
            
    ints_score=ints_val/(m*n)
    
    var_val=[]
    #clarity evaluation of the image
    for i in range(0,m-block_row_size,block_row_size):
        for j in range(0,n-block_coloumn_size,block_coloumn_size):
            img_block = img[i:i+block_row_size, j:j+block_coloumn_size]
            block_var = np.var(img_block)
            var_val.append(block_var)
    # print (var_val)
    var_score=[]    
    for k in (var_val):
        if(k<=variance_thresold):
            var_score.append(k)    #removing higher variance 
            
    clarity_score=np.mean(var_score)       #clarity score
    

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # s=1
    # if s==1:
    #     print (thresh)
    #     s+=1
    #integrity calculation  
    B=[[0 for i in range(n)]for j in range(m)]
    for i in range(m):
        for j in range(n):
            val=(i-m/2)**2+(j-n/2)**2
            if(val<((d1*n)**2)):
                B[i][j]=1
            else:
                B[i][j]=0
    intg_val=0      
    for i in range(m):
        for j in range(n):
            intg_val+=B[i][j]     # intigrity_value
            
    intg_score=intg_val/(m*m*n*n)
    ls=[]
    ls.append(ints_score)
    ls.append(clarity_score)
    ls.append(intg_score)

    return ls

def pupil_extraction(img):

    gray=img.copy()
    # cv2.imshow('i',img)
    cv2.waitKey(0)

  

    # Define the gray ranges for each region
    # pupil_range = [0, 60]
    # iris_range = [30, 80]
    # sclera_range = [80, 150]
    # eyelash_range = [150, 255]

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# The gray ranges for each region for CASIA
    pupil_range = [0, 50]
    iris_range = [30, 80]
    sclera_range = [80, 150]
    eyelash_range = [150, 200]

    #Applying adaptive binary threshold segmentation to extract each region
    pupil = cv2.inRange(gray, pupil_range[0], pupil_range[1])
    iris = cv2.inRange(gray, iris_range[0], iris_range[1])
    sclera = cv2.inRange(gray, sclera_range[0], sclera_range[1])
    eyelash = cv2.inRange(gray, eyelash_range[0], eyelash_range[1])

    #Removing noise
    kernel = np.ones((5, 5), np.uint8)
    pupil = cv2.erode(pupil, kernel, iterations=1)
    pupil = cv2.dilate(pupil, kernel, iterations=1)
    pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, kernel)

    iris = cv2.erode(iris, kernel, iterations=1)
    iris = cv2.dilate(iris, kernel, iterations=1)
    iris = cv2.morphologyEx(iris, cv2.MORPH_CLOSE, kernel)

    sclera = cv2.erode(sclera, kernel, iterations=1)
    sclera = cv2.dilate(sclera, kernel, iterations=1)
    sclera = cv2.morphologyEx(sclera, cv2.MORPH_CLOSE, kernel)

    eyelash = cv2.erode(eyelash, kernel, iterations=1)
    eyelash = cv2.dilate(eyelash, kernel, iterations=1)
    eyelash = cv2.morphologyEx(eyelash, cv2.MORPH_CLOSE, kernel)

    # Combineing the iris, sclera, and eyelash regions to extract the pupil region
    mask = iris + sclera + eyelash
    
    pupil = cv2.bitwise_and(pupil, mask)

    contours, hierarchy = cv2.findContours(pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # the center and radius of the pupil
    (x,y), radius = cv2.minEnclosingCircle(max_contour)
    center = (int(x), int(y))
    radius = int(radius)


    return center,radius


def ContourIntegralCircular(imagen, y_0, x_0, r, angs):

    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)


    ind = np.where(x < 0)
    x[ind] = 0
    rows=imagen.shape[1]
    ind = np.where(x >= rows)
    x[ind]=rows-1
    ind = np.where(y < 0)
    y[ind] = 0
    cols=imagen.shape[0]
    ind = np.where(y >= cols)
    y[ind] = cols - 1
    

    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    ans=hs.astype(float)
    return ans




def iris_outer(img, inner_y, inner_x, inner_r):
    
    minrad=inner_r*1.3
    maxrad=inner_r*4
    maxdispl=inner_r*0.15
    maxdispl = np.round(maxdispl).astype(int)
    d_maxdispl=2*maxdispl
    pi_val=np.pi
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * pi_val
    minrad = np.round(minrad).astype(int)
    maxrad = np.round(maxrad).astype(int) 

    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    
    y,x,r = inner_y - maxdispl + y,inner_x - maxdispl + x,minrad + r

    hs = ContourIntegralCircular(img, y, x, r, angs)
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]
    sm = 13 	
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)
    outer_y,outer_x, outer_r = inner_y - maxdispl + y + 1,inner_x - maxdispl + x + 1,minrad + r - 1

    return outer_y, outer_x, outer_r
