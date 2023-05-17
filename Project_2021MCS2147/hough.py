import numpy as np
import os
import cv2 
from utills import *
import time

import csv
import itertools
import math
from typing import Tuple, List

def show_img(ciriris,cirpupil,img1,img_name):
    i_x=ciriris[1]
    i_y=ciriris[0]
    i_r=ciriris[2]

    p_x=cirpupil[1]
    p_y=cirpupil[0]
    p_r=cirpupil[2]
    cv2.circle(img1,(i_x, i_y), i_r, (255, 255,255), 2)
    cv2.circle(img1,(p_x, p_y), p_r, (255, 255,255), 2)
    # fname='outputd1/'+img_name
    # cv2.imwrite(fname,img1)
    #cv2.imshow('img',img1)
    cv2.waitKey(0)


def daugman(gray_img: np.ndarray, center: Tuple[int, int],
            start_r: int, end_r: int, step: int = 1) -> Tuple[float, int]:
    x, y = center
    intensities = []
    mask = np.zeros_like(gray_img)

    radii = list(range(start_r, end_r, step)) 
    for r in radii:
        cv2.circle(mask, center, r, 255, 1)
        diff = gray_img & mask
        intensities.append(np.add.reduce(diff[diff > 0]) / (2 * math.pi * r))
        mask.fill(0)

    intensities_np = np.array(intensities, dtype=np.float32)
    del intensities


    intensities_np = intensities_np[:-1] - intensities_np[1:]

    intensities_np = abs(cv2.GaussianBlur(intensities_np, (1, 5), 0))

    idx = np.argmax(intensities_np)  # type: int


    return intensities_np[idx], radii[idx]


def find_iris(gray: np.ndarray, *,
              daugman_start: int, daugman_end: int,
              daugman_step: int = 1, points_step: int = 1,) -> Tuple[Tuple[int, int], int]:

    h, w = gray.shape
    single_axis_range = range(int(h / 3), h - int(h / 3), points_step)
    all_points = itertools.product(single_axis_range, single_axis_range)

    intensity_values = []
    coords = []  

    for point in all_points:
        val, r = daugman(gray, point, daugman_start, daugman_end, daugman_step)
        intensity_values.append(val)
        coords.append((point, r))
    best_idx = intensity_values.index(max(intensity_values))
    return coords[best_idx]

   
dic={}
l1=0
with open('data1.csv', mode ='r')as file:
# reading the CSV file
    csvFile = csv.reader(file)
    
    # displaying the contents of the CSV file
    for lines in (csvFile):
            # print(lines)
        if l1%2==1:
            l1+=1
            continue
        l1+=1
        if len(lines)!=0:
            ls=[]
            a=str(lines[0])
            for i in range(1,len(lines)):
                ls.append(float(lines[i]))
            dic[a]=ls
# print (dic)
res1,res2=0,0
th=5.0
total_val=0
dataset_path1 = "CASIA2"
filename = "data4.csv"
list1 = []
names=[]
with open(filename, 'w') as csvfile:
    for root, dirs, files in os.walk(dataset_path1):
        st=time.time()
        for file in files:
            total_val+=1
            if file.endswith(".jpg"):

                img = cv2.imread(os.path.join(root, file),0)
                img=cv2.medianBlur(img, 5)
                
                #img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
                f1,f2=0,0
                coords = find_iris(img, daugman_start=20, daugman_end=150, daugman_step=2, points_step=4)
                center,rp=coords
                colp, rowp=center 
                row, col, r = searchOuterBound(img, rowp, colp, rp)

                
                rowp = np.round(rowp).astype(int)
                colp = np.round(colp).astype(int)
                rp = np.round(rp).astype(int)
                row = np.round(row).astype(int)
                col = np.round(col).astype(int)
                r = np.round(r).astype(int)
                cirpupil = [rowp, colp, rp]
                ciriris = [row, col, r]
                #ciriris, cirpupil = segment(img, 70)
                show_img(ciriris,cirpupil,img,file)
                xp,yp,rp,xi,yi,ri=cirpupil[0],cirpupil[1],cirpupil[2],ciriris[0],ciriris[1],ciriris[2]
        et=time.time()
    
print("Average time for iris Localization",(et-st)/total_val)





