import cv2,os
import numpy as np
from register import *
from utills import *
import math
import time


import csv
ints_thres=85
clarity_thres=1.1
integrity_tres=0.025

    # cv2.imshow('img',img1)
    #cv2.waitKey(0)

def scaleup(x,factor):
    ans=int((x/factor))
    return ans+1

def show_img(ciriris,cirpupil,img1,img_name):
    i_x=ciriris[1]
    i_y=ciriris[0]
    i_r=ciriris[2]

    p_x=cirpupil[1]
    p_y=cirpupil[0]
    p_r=cirpupil[2]
    cv2.circle(img1,(i_x, i_y), i_r, (255, 255,255), 2)
    cv2.circle(img1,(p_x, p_y), p_r, (255, 255,255), 2)
    fname='output/'+img_name
    cv2.imwrite(fname,img1)
    #cv2.imshow('img',img1)
    cv2.waitKey(0)

def Iris_Localization(img1,img_name,factor=0.6):

    row1,col1=img1.shape
    square_size=min(row1,col1)
    img1=cv2.medianBlur(img1, 5)

    #img1=cv2.resize(img1, (320, 240), interpolation = cv2.INTER_LINEAR)
    #
    
    img1_show=img1.copy()
    #cv2.imshow("gausiian", img1_show) 
    cv2.medianBlur(img1,5)
    cv2.waitKey(0)



    img1 = cv2.GaussianBlur(img1, (13,13), 2)


    #gamma correction
    gamma = 1.1
    img1 = np.power(img1 / 255.0, gamma)
    img1 = np.uint8(img1 * 255)


    # cv2.imshow("Gamma Correction", img1)
    cv2.waitKey(0)


    # calculate the differences in vertical direction
    d_vert = np.abs(np.diff(img1, axis=0))

    # calculate the differences in horizontal direction
    d_horiz = np.abs(np.diff(img1, axis=1))

    # create kernels for oblique and anti-oblique differences
    kernel_oblique = np.array([[-1 ,0], [0, 1]])
    kernel_anti_oblique = np.array([[1,0], [0,-1]])

    d_oblique = np.abs(cv2.filter2D(img1, -1, kernel_oblique))
    d_anti_oblique = np.abs(cv2.filter2D(img1, -1, kernel_anti_oblique))
    min_rows=np.min([d_vert.shape[0],d_horiz.shape[0],d_oblique.shape[0],d_anti_oblique.shape[0]])
    min_cols=np.min([d_vert.shape[1],d_horiz.shape[1],d_oblique.shape[1],d_anti_oblique.shape[1]])
    d_x=[[0 for i in range(min_cols)]for i in range(min_rows)]
    d_y=[[0 for i in range(min_cols)]for i in range(min_rows)]
    direction=[[0 for i in range(min_cols)]for i in range(min_rows)]
    gradient=[[0 for i in range(min_cols)]for i in range(min_rows)]
    for i in range(min_rows):
        for j in range(min_cols):
            d_x[i][j]=d_vert[i][j]+(d_oblique[i][j]+d_anti_oblique[i][j])/2
            d_y[i][j]=d_vert[i][j]+(d_oblique[i][j]-d_anti_oblique[i][j])/2
            gradient[i][j]=d_x[i][j]*i+d_y[i][j]*j
            direction[i][j]=np.arctan(-(d_y[i][j])/d_x[i][j])



    # perform non-maximum suppression to extract edge points
    height, width = min_rows,min_cols
    edge_points = []
    for i in range(1, height-1):
        for j in range(1, width-1):
            # check the direction of the gradient
            if direction[i][j] >= -np.pi/8 and direction[i][j] < np.pi/8:
                if gradient[i][j] > gradient[i][j-1] and gradient[i][j] > gradient[i, j+1]:
                    edge_points.append((i, j))
            elif direction[i][j] >= np.pi/8 and direction[i][j] < 3*np.pi/8:
                if gradient[i][j] > gradient[i-1, j+1] and gradient[i][j] > gradient[i+1, j-1]:
                    edge_points.append((i, j))
            elif direction[i][j] >= 3*np.pi/8 or direction[i][j] < -3*np.pi/8:
                if gradient[i][j] > gradient[i-1][j] and gradient[i][j] > gradient[i+1][ j]:
                    edge_points.append((i, j))
            elif direction[i][j] >= -3*np.pi/8 and direction[i][j] < -np.pi/8:
                if gradient[i][j] > gradient[i-1][j-1] and gradient[i][j] > gradient[i+1][j+1]:
                    edge_points.append((i, j))

    edges=[]
    # draw the detected edge points on the input image
    for point in edge_points:

        if(img1[point[0]][point[1]]in range(140,150)):
            #cv2.circle(img1, point[::-1], 3, (255, 255,255), 1)
            edges.append([point[0],point[1]])
    


    center,radius=pupil_extraction(img1)

    cv2.circle(img1_show, center, radius, (255, 255, 0), 2)

    # cv2.imshow("Pupil ", img1_show)
    cv2.waitKey(0)

   

    #edges=cv2.Canny(img1,100,200)
    outer_y, outer_x, outer_r=iris_outer(img1,center[1],center[0],radius)
    

    #outer_y, outer_x, outer_r=iris_outer(img1,center[0],center[1],radius)
    cv2.circle(img1_show,center, outer_r, (255, 255,255), 2)
    fname='output/'+img_name
    cv2.imwrite(fname,img1_show)
    #cv2.imshow('img',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    # print(center,radius)
    # v={}
    # for point in edges:
    #     e=(point[0]-center[0])**2+(point[1]-center[1])**2-radius**2
    #     if(abs(e)<10):

    #        k=10

    #print(v)
    return center[0],center[1],radius,outer_x, outer_y, outer_r



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
res1,res2,res3=0,0,0
th=8.0
total_val=0
dataset_path1 = "CASIA1"
list1 = []
names=[]
k=0  # counting for poor image
for root, dirs, files in os.walk(dataset_path1):
    st=time.time()
    for file in files:
        total_val+=1
        if file.endswith(".jpg"):
            # reading the image 
            img = cv2.imread(os.path.join(root, file),0)

            img1=img.copy()
            
            f1,f2=0,0
            intensity_val,clarity_val,integrity_val=image_Quality(img,200,0.20)

            if(intensity_val>ints_thres and clarity_val>clarity_thres):
                #print("image quality is accepted")
                cb=10
            else:
                #print("image quality is Poor")
                k+=1
                continue
            factor=0.8
            img=cv2.resize(img1, (0,0), fx=factor, fy=factor)
            yp,xp,rp,yi,xi,ri=Iris_Localization(img,file,factor)
            # scaling 

            yp,xp,rp,yi,xi,ri=scaleup(yp,factor),scaleup(xp,factor),scaleup(rp,factor),scaleup(yi,factor),scaleup(xi,factor),scaleup(ri,factor)


            cirpupil = [xp, yp, rp]
            ciriris = [xi, yi, ri]
            show_img(ciriris,cirpupil,img1,file)
            output = (str(os.path.basename(file)))[0:12]
            center_list=dic[output]
            #print (xp,yp,rp,xi,yi,ri,center_list,total_val)
            cirpupil = [yp, xp, rp]
            ciriris = [yi,xi,ri]

            if abs(xp-center_list[0])<=th and abs(yp-center_list[1])<=th and abs(rp-center_list[2]<=th):
                res1+=1
                f1=1
            
            if abs(xi-center_list[3])<=th and abs(yi-center_list[4])<=th and abs(ri-center_list[5]<=th):
                res2+=1
                f2=1
            if(f1==1 and f2==1):
                res3+=1
    et=time.time()
    
print("Average time for iris Localization",(et-st)/total_val)
print ("Pupil localization Accuracy : ", (res1/(total_val-k) * 100 ))
print ("Iris localization Accuracy : ", (res2/(total_val-k) * 100 ))
print ("Iris localization Accuracy : ", (res3/(total_val-k) * 100 ))

            






    





