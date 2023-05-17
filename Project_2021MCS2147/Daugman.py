import numpy as np
import os
import cv2 
from utills import *

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
    fname='outputd2/'+img_name
    cv2.imwrite(fname,img1)
    #cv2.imshow('img',img1)
    cv2.waitKey(0)


   
dic={}
l1=0

res1,res2=0,0
th=5.0
total_val=0
dataset_path1 = "CASIA1"
filename = "data2.csv"
list1 = []
names=[]
with open(filename, 'w') as csvfile:
    for root, dirs, files in os.walk(dataset_path1):
        for file in files:
            total_val+=1
            if file.endswith(".jpg"):
                
                
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                
                img = cv2.imread(os.path.join(root, file),0)
                img=cv2.medianBlur(img, 5)
                img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)
                f1,f2=0,0
                

                rowp, colp, rp = searchInnerBound(img)
                row, col, r = searchOuterBound(img, rowp, colp, rp)
                rowp = np.round(rowp).astype(int)
                colp = np.round(colp).astype(int)
                rp = np.round(rp).astype(int)
                row = np.round(row).astype(int)
                col = np.round(col).astype(int)
                r = np.round(r).astype(int)
                cirpupil = [rowp, colp, rp]
                ciriris = [row, col, r]
                
                show_img(ciriris,cirpupil,img,file)
                xp,yp,rp,xi,yi,ri=cirpupil[0],cirpupil[1],cirpupil[2],ciriris[0],ciriris[1],ciriris[2],
                
                # writing the fields 
                rows=[file,cirpupil[0],cirpupil[1],cirpupil[2],ciriris[0],ciriris[1],ciriris[2]]
                csvwriter.writerow(rows) 
                output = (str(os.path.basename(file)))[0:8]

           



