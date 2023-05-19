import cv2
import numpy as np
import fingerprint_enhancer
import math
import skimage
from skimage.morphology import skeletonize, thin
import cv2 
import sys
path1=sys.argv[1]
path2=sys.argv[2]
img = cv2.imread(path1,0)
img2 = cv2.imread(path2,0)
cv2.imshow("orignal",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def neighbors(angles,i,j): 
    lst2=[] 
    p=i-1
    for k in range(-1,1):
        lst2.append(angles[p][j+k])
    q=j+1
    for k in range(-1,2):
        lst2.append(angles[i+k][q])
    p=i+1
    for k in range(0,-1,-1):
        lst2.append(angles[p][j+k])
    q=j-1
    for k in range(1,-2,-1):
        lst2.append(angles[i+k][q])
    return lst2

def rounding(G,l,k):
    a=G[l,k]
    a=round(a)
    return a

def rounding(G,l,k):
    a=G[l,k]
    a=round(a)
    return a
def call(n,m,res,j,w):
    p1=j-1
    pos=int(p1//w)
    pi=math.pi
    tan=math.atan2(n,m)
    angle=(pi+tan)/2
    if n==0 and m==0:
        res[pos].append(0)
    else:
        res[pos].append(angle)
    return res
def calculate_angles(im, W,n=0,m=0):

    (y, x) = im.shape
    sobel_list_x,sobel_list_y=[],[]
    sobel_list_x.append([-1,0,1])
    sobel_list_x.append([-2, 0, 2])
    sobel_list_x.append([-1,0,1])
    sobel_list_y.append([-1,-2,-1])
    sobel_list_y.append([0,0,0])
    sobel_list_y.append([1,2,1])
    sobel_list_x,sobel_list_y=np.array(sobel_list_x,np.int8),np.array(sobel_list_y,np.int8)
    temp=sobel_list_x
    sobel_list_x=sobel_list_y
    sobel_list_y=temp
    result,Gx_,Gy_ = [[] for i in range(1, y, W)],cv2.filter2D(im/125,-1, sobel_list_y)*125,cv2.filter2D(im/125,-1, sobel_list_x)*125
    for j in range(1, y, W):
        for i in range(1, x, W):
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W , x - 1)):
                    a=rounding(Gx_,l,k)
                    b=rounding(Gy_,l,k)
                    n,m=n+(2*a*b),m+(a**2-b**2)
            result=call(n,m,result,j,W)
            n,m=0,0
    result=np.array(result)
    return result

def segmented(skel1):
    (y, x) = skel1.shape
    
    im=skel1.copy()
    segmented_image=im.copy()
    image_variance = np.zeros((y,x))
    threshold=math.sqrt(abs(np.mean(im-np.mean(im))))*0.2
    w=4*4
    for i in range(0, x, 16):
        for j in range(0, y, 16):
            row1,row2=i,min(x,i+w)
            col1,col2=j,min(y,j+w)
            box = [row1,col1,row2, col2]
            blk=im[box[1]:box[3], box[0]:box[2]]
            blk_std = np.std(blk)
            image_variance[box[1]:box[3], box[0]:box[2]] = blk_std
    mask=np.ones_like(im)
    mask[image_variance<threshold]=0
    r1=w*2
    c1=w*2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r1, c1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    segmented_image *= mask
    return segmented_image,mask

def calculate_minutiaes(im,angles,wnd_size):
    mnt=[]
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)
    #biniry_image = biniry_image.astype(np.int8)
    (y, x) = im.shape
    result = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    colors = {1 :(150,0,0), 3 : (0, 150, 0)}
    t=wnd_size//2
    M=x-t
    N=y-t
    for i in range(1,M):
        for j in range(1,N):
            if biniry_image[j][i] == 1:
               values = neighbors(biniry_image,j,i) 
               crossings = 0
               for k in range(0, len(values)-1):
                    dif=abs(values[k] - values[k + 1])
                    crossings=crossings+dif
                    #crossings+=values[k]
               crossings = crossings// 2
           
                #print("hjjjkj")
                
               if(crossings==1):
                    mnt.append((i,j,angles[(j-1)//wnd_size][(i-1)//wnd_size],1))
                    cv2.circle(result, (i,j), radius=2,color= (255, 0, 0), thickness=-1)
               elif(crossings==3):
                    mnt.append((i,j,angles[(j-1)//wnd_size][(i-1)//wnd_size],2))
                    cv2.circle(result, (i,j), radius=2, color=(0, 0, 255), thickness=-1)
               
                
    return result,mnt






def check_core(angle_at, tolerance,angles2):
    index=0
    for k in range(8):
        # calculate the difference
        dif=angle_at[k+1]-angle_at[k]
        dif2=dif-180
        dif3=dif+180
        if -90<=dif<=90 :
            index+=dif
        elif -90<=dif2<=90:
            index+=dif2
        else:
            index+=dif3

    #1 for loop #2 for delta #3 for whorl #4 for normal
    if(abs(180-index)<=tolerance): 
        return 1
    if(abs(-180-index)<=tolerance):
        return 2
    if(abs(360-index)<=tolerance):
        return 3
    return 4


def calculate_singularities(im,angles,tolerance,W,mask):
    result =cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    M=len(angles)
    N=len(angles[0])
    angles2=angles.copy()
    for i in range(M):
        for j in range(N):
            angles2[i][j]=math.degrees(angles[i][j])
    core_points=[]   
    for i in range(3, M - 2):             
        for j in range(3, N - 2):      
            rows1,rows2=(i-2)*W,(i+3)*W
            col1,col2=(j-2)*W,(j+3)*W,
            mask_slice=mask[rows1:rows2,col1:col2]
            
            
            mask_flag=np.sum(mask_slice)
            lmt=(W*5)*(W*5)
            if mask_flag==lmt:
                angle_at=neighbors(angles2,i,j)
                is_core=check_core(angle_at, tolerance,angles2)
                row=j*W
                col=i*W
                if is_core==1:
                    core_points.append((j*W,i*W,angles[i][j]))
                    cv2.circle(result,(  row, col), radius=2,color= (100,149,237),thickness= 3)
                if is_core==2:
                    core_points.append((j*W,i*W,angles[i][j]))
                    cv2.circle(result,  (row, col), radius=2,color= (255,140,0),thickness= 3)
                if is_core==3:
                    core_points.append((j*W,i*W,angles[i][j]))
                    cv2.circle(result,( row, col), radius=2,color= (193,255,193), thickness=3)
    return result,core_points

#transforming with respect to core point , mnt contains x,y,theta and type Bifurication or endpoint
def trnsfwrtcore(mnt,crx,cry,crtheta): 
    lst=[]
    type=0
    for i in range(len(mnt)):
        de1=math.sqrt((crx-mnt[i][0])**2+(cry-mnt[i][1])**2)
        temp=abs(crtheta-mnt[i][2])
        theta=min(temp,360-temp)
        
        lst.append((de1,theta,mnt[i][3]))
    return lst

# use trnsfunction to generate tmnt;

#tmnt contain sd1,orienttaion with respect to core point and type of point #function returns no of point matchded
def matchingcore(tmnt1,tmnt2,min_dist,min_theta):
    M=len(tmnt1)
    N=len(tmnt2)
    f1=[0 for i in range(M)]
    f2=[0 for i in range(N)]
    res=[]
    count=0
    for i in range(M):
        for j in range(N):
            if(f1[i]==0 and f2[j]==0 and tmnt1[i][2]==tmnt2[j][2]):
                d_dis=abs(tmnt1[i][0]-tmnt2[j][0])
                d_theta=abs(tmnt1[i][1]-tmnt2[j][1])
                if(d_dis<=min_dist and d_theta<=min_theta):
                    count+=1
                    f1[i]=1
                    f2[j]=1

    #print(count)
    return (float(count)/min(M,N)*100)



#testing
#img=normalize(input_img.copy(),float(100),float(100))

def thinning(img):
    skel=[]
    ar = np.uint8(img)
    ar=img>128
    skel=np.uint8(skimage.morphology.skeletonize(ar))*255
    return skel

eimg1= fingerprint_enhancer.enhance_Fingerprint(img)
eimg2= fingerprint_enhancer.enhance_Fingerprint(img2)

skel1=~thinning(eimg1)
cv2.waitKey(0)
skel2=~thinning(eimg2)
cv2.imshow("skel",skel1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("skel2",skel2)
cv2.waitKey(0)
cv2.destroyAllWindows()



#angles=calculate_angles(skel1,3)
#print(angles.size,angles)
segmented_image, mask= segmented(skel1)

segmented_image2, mask2= segmented(skel2)

#orientation_img = visualize_angles(segmented_image, mask, angles, W=3)
angles1=calculate_angles(skel1,3)
angles2=calculate_angles(skel2,3)

#cv2.imshow("Oriented",orientation_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

result,mnt1=calculate_minutiaes(skel1,angles1,3)
result12,mnt2=calculate_minutiaes(skel2,angles2,3)
cv2.imshow("minutae 1",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("minutae 2",result12)
cv2.waitKey(0)
cv2.destroyAllWindows()

angles1=calculate_angles(skel1,16)
angles2=calculate_angles(skel2,16)

result2,core_Points2 = calculate_singularities(skel1, angles1, 2, 16,mask)
result22,core_Points22 = calculate_singularities(skel2, angles2, 2, 16,mask2)
cv2.imshow("minutae",result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("minutae",result22)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(core_Points2,core_Points22)
ans=0.00000
for i in core_Points2:
    #print(i,type(i))
    tmnt1=trnsfwrtcore(mnt1,i[0],i[1],i[2])
    for j in core_Points22:
        tmnt2=trnsfwrtcore(mnt2,j[0],j[1],j[2])
        ans=max(ans,matchingcore(tmnt1,tmnt2,12,0.21))
        #print("some answer",ans)   
print(ans)




