import numpy as np
import csv
import pandas as pd
from scipy.ndimage import gaussian_filter
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def sig_verify(fname1,fname2):
    def Pre_Process(fname):
        #Pre Processing 
        #1.Center of Mass calculation
        
        dset=[]
        normalize_dset=[]
        sum_x=0.00000
        sum_y=0.0000
        center_mass_x=0.000000
        center_mass_y=0.000000
        intial_time=0
        final_time=0
        flag=1

        with open(fname, 'r') as f:
            for line in f:
                l=[]

                words = line.split()
                count=1
                for i in words:
                    l.append(int(i))
                    if(count==1):
                        sum_x+=int(i)
                    elif(count==2):
                        sum_y+=int(i)
                    elif(flag==1 and count==3):
                        intial_time=int(i)
                        flag+=1
                    elif(count==3):
                        final_time=int(i)
                    
                    count=count+1
                
                
                dset.append(l)

        dset.pop(0)

        #print(dset)
        total_time=final_time-intial_time
        center_mass_x=sum_x/total_time
        center_mass_y=sum_y/total_time

        #2.Normalize the Coordinates
        for i in range(len(dset)):
            #dset[i][0]-=
            dset[i][0]-=center_mass_x
            dset[i][1]-=center_mass_y

        gaussian_filter(dset, sigma=1)
        return dset

    def critical_speed(dset):
        c=1
        lst=[]
        intial_pos=dset[0][1]
        intial_time=dset[0][2]
        f=1
        
        for i in range(1,len(dset)):
            if(dset[i][1]>dset[i-1][1] and f== -1 ):
                final_pos=dset[i][1]
                final_time=dset[i][2]
                div=max((final_time-intial_time),0.00001)
                vs=(final_pos-intial_pos)/div
                intial_pos=final_pos
                intial_time=final_time
                for j in range(c):
                    lst.append(vs)
                f= 1
                c=1
            elif(dset[i][1]<dset[i-1][1] and f== 1):
                final_pos=dset[i][1]
                final_time=dset[i][2]
                vs=(final_pos-intial_pos)/(final_time-intial_time)
                intial_pos=final_pos
                intial_time=final_time
                for j in range(c):
                    lst.append(vs)
                f= -1
                c=1
            else:
                c+=1
        final_time=dset[-1][2]
        final_pos=dset[-1][1]
        div=max((final_time-intial_time),0.00001)
        vs=(final_pos-intial_pos)/div
        for k in range(c):
            
            lst.append(vs)
        #print(len(lst))
        return lst


    #Feature Extraction
    def feature_extraction(dset):
        f_set=[]        #0 for i in range(len(dset[0]))] for j in range(len(dset))]
        x_prev=0
        y_prev=0
        t_prv=0
        for i in range(len(dset)):
            if(i==0):
                lst=[0.00,0.00,0.00,0.000,0.0000,0.0000,0.0000,0.00,0.000,0.0000,0.000,0.000000]
            else:
                lst=[]
        
                lst.append(dset[i][6])
                lst.append(dset[i][5])
                lst.append(dset[i][4])
                tm_gap=max(dset[i][2]-dset[i-1][2],0.0001)
                vx=dset[i][0]-dset[i-1][0]/tm_gap
                vy=dset[i][1]-dset[i-1][1]/tm_gap
                v=math.sqrt(vx*vx+vy*vy)


                lst.append(abs(vx))
                lst.append(abs(vy))
                lst.append(v)
                if(i!=1):
                    ax=lst[3]-f_set[i-1][3]/tm_gap
                    ay=lst[4]-f_set[i-1][4]/tm_gap
                else:
                    ax=0.00000000
                    ay=0.00000000
                #accelaration=math.sqrt(ax*ax+ay*ay)
                lst.append(ax)
                lst.append(ay)
                #lst.append(accelaration)

                
                dpt=dset[i][6]-dset[i-1][6]/tm_gap
                div=max((dset[i][0]-dset[i-1][0]),0.0000001)
                dxt=dset[i][6]-dset[i-1][6]/div
                div=max((dset[i][1]-dset[i-1][1]),0.0000001)
                dyt=dset[i][6]-dset[i-1][6]/div
                lst.append(dpt)
                lst.append(dxt)
                lst.append(dyt)
        
            
            f_set.append(lst)


        
        
        f_set.pop(0)
        crt_speed=critical_speed(dset)

        for i in range(0,len(f_set)):
            f_set[i].append(crt_speed[i])

        return f_set

    # feature normalization
    def feature_normalization(fset):
        scale= StandardScaler()
        fset = scale.fit_transform(fset) 
        #print(fset.shape)
        return fset

    dset1=Pre_Process(fname1)
    #print(dset1)
    dset2=Pre_Process(fname2)
    fset1=feature_extraction(dset1)
    fset2=feature_extraction(dset2)
    #print(fset1)
    #print(critical_speed(dset1))
    fset1=feature_normalization(fset1)
    fset2=feature_normalization(fset2)

    # cost matrix
    def feature_cost_matrix(St,Qt):
        #st[i] sample matrix qt[i]  query matrix
        m=len(St)
        n=len(Qt)
        d=[[0 for i in range(n)] for j in range(m)]         # m*n cost matrix

        for i in range(m):
            for j in range(n):
                dis=0.00000
                #for k in range(len(St[0])):
                dis+=abs(St[i]-Qt[j]) 
                d[i][j]= dis

        return d             

    # DTW matrix calculation using dynamic programing
    def DTWmatrix(Costmatrix,Dmatrix):
        M=len(Costmatrix)
        N=len(Costmatrix[0])
        Dmatrix[0][0]=Costmatrix[0][0]
        for j in range(1,len(Costmatrix[0])):
            Dmatrix[0][j]=Costmatrix[0][j]+Dmatrix[0][j-1]
        for i in range(1,len(Costmatrix)):
            Dmatrix[i][0]=Costmatrix[i][0]+Dmatrix[i-1][0]


        for i in range(1,M):
            for j in range(1,N):
                Dmatrix[i][j]=Costmatrix[i][j]+min(min(Dmatrix[i][j-1],Dmatrix[i-1][j]),Dmatrix[i-1][j-1])

        return Dmatrix

    feature_DTW=[]
    for i in range(12):
        
        cst_matrix=feature_cost_matrix(fset1[i],fset2[i])
        m=len(cst_matrix)
        n=len(cst_matrix[0])
        fd=[[0 for i in range(n)] for j in range(m)]
        dist=DTWmatrix(cst_matrix,fd)
        feature_distance=dist[-1][-1]
        feature_DTW.append(feature_distance)

    
    #print("skmdfl;knas;knf;aksn;fkan;fsnf ",feature_DTW,"skfmlkaml")
    return feature_DTW

def calculate_weight():
    result=[]
    X=[]
    y=[]
    entries1 = Path('dataset1/')
    i=0
    for entry1 in entries1.iterdir():
        j=0
        entries2 = Path('dataset1/')
        
        for entry2 in entries2.iterdir():
            lst=(sig_verify(entry1,entry2))
            lst2=lst.copy()
            X.append(lst2)
            #print('This is before',lst)
            if((i//10)==(j//10)):
                lst.append(1)
                y.append(1)
            else:
                lst.append(0)
                y.append(0)
            result.append(lst)

            #print("hak")
            j+=1
        i+=1
        
        #print("\n")    
    # for i in result:
    #     print(i)
    filename = "signature_data.csv"
    with open(filename, 'w') as csvfile: 
    
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerows(result)


    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    #print(X_train,X_test,y_train,y_test)


    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=60, learning_rate=0.9, random_state=0)
    
    model1 = model.fit(X_train, y_train)

    feature_weight= model1.feature_importances_
    #print(feature_weight)

    y_pred = model1.predict(X_test)
    from sklearn.metrics import accuracy_score


    # calculate and print model accuracy
    print("AdaBoostModel accuracy:",accuracy_score(y_test, y_pred))
    features_dict = {
        'pressure': feature_weight[0],
        'altitude': feature_weight[1],
        'azimuth': feature_weight[2],
        'vx': feature_weight[3],
        'vy': feature_weight[4],
        'speed': feature_weight[5],
        'Ax': feature_weight[6],
        'Ay': feature_weight[7],
        'dpt': feature_weight[8],
        'dpx': feature_weight[9],
        'dpy': feature_weight[10],
        'critical': feature_weight[11]
    }

    # Sort the features based on their score in descending order
    sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted features
    for feature in sorted_features:
        print(feature[0], feature[1])
    
    return feature_weight,model1


def Similarity_Score(feature_weights,feature_distance):
    disim_score=0.000000
    for i in range(len(feature_distance)):
        disim_score+=feature_weights[i]*feature_distance[i]
        #print("disimilarity",disim_score)
    
    return disim_score

fname1 = './dataset/USER1_1.txt'
fname2 = './dataset/USER3_1.txt'
lst=(sig_verify(fname1,fname2))
feature_weights,trained_model=calculate_weight()

print("Weighted Similarity Score",Similarity_Score(feature_weights,lst))
y_pred = trained_model.predict([lst])

if(y_pred==1):
    print("Signature is mathced")
else:
    print("signature not matched")

