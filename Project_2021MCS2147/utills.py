import numpy as np
import numpy as np
from scipy.ndimage import convolve
from scipy import signal
from skimage.transform import radon

def jump_val(inner_x,jump,x):
    ans=x=inner_x-jump+x
    return ans

def searchInnerBound(img):
    Y,X = img.shape[0:2]
    
    sect=X/4
    minrad=10
    maxrad=sect*0.8 		
    d_pi=2*np.pi
    d_sect=2*sect
    jump=4 	


    # Hough Space (y,x,r)
    sz=np.array([np.floor((Y-d_sect)/4),np.floor((X-d_sect)/4),np.floor((maxrad-minrad)/jump)]).astype(int)

    # Resolution of the circular integration
    integrationprecision = 1
    angs= np.arange(0, d_pi, integrationprecision)
    x,y,r= np.meshgrid(np.arange(sz[1]),  np.arange(sz[0]),np.arange(sz[2]))
    y=sect + y*4
    x=sect + x*4
    r=minrad + r*4
    hs=ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative R
    hs_size=hs.shape[2]-1
    hspdr=hs-hs[:, :, np.insert(np.arange(hs_size), 0, 0)]

    # Blur
    sm=3 		# Size of the blurring mask
    hspdrs=signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax=np.argmax(hspdrs.ravel())
    y,x,r=np.unravel_index(indmax, hspdrs.shape)

    inner_y=sect + y*4
    inner_x=sect + x*4
    inner_r=minrad + (r-1)*4

    # Integro-Differential operator fine (pixel-level precision)
    integrationprecision = 0.1 
    		# Resolution of the circular integration
    angs=np.arange(0, d_pi, integrationprecision)
    x,y,r= np.meshgrid(np.arange(8),np.arange(8),np.arange(8))
    y=jump_val(inner_y,4,y)
    x=jump_val(inner_x,4,x)
    r=jump_val(inner_r,4,r)
    hs=ContourIntegralCircular(img, y, x, r, angs)
    hs_size1=hs.shape[2]-1
   
    hspdr=hs-hs[:, :, np.insert(np.arange(hs_size1), 0, 0)]

    # Bluring
    sm=3 		# Size of the blurring mask
    hspdrs=signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax=np.argmax(hspdrs.ravel())
    y,x,r=np.unravel_index(indmax, hspdrs.shape)

    inner_y=jump_val(inner_y,4,y)
    inner_x=jump_val(inner_x,4,x)
    inner_r=jump_val(inner_r,4,r-1)

    return  int(inner_y), int(inner_x), int(inner_r)


def searchOuterBound(img, inner_y, inner_x, inner_r):
    
    minrad=inner_r/0.8
    maxrad=inner_r/0.3
    maxdispl=inner_r*0.15
    maxdispl = np.round(maxdispl).astype(int)
    d_maxdispl=2*maxdispl
    pi_val=np.pi
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * pi_val
    minrad = np.round(minrad).astype(int)
    maxrad = np.round(maxrad).astype(int)   
    
    # Resolution of the circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(d_maxdispl),np.arange(d_maxdispl), np.arange(maxrad-minrad))
    y = jump_val(inner_y ,maxdispl ,y)
    x = jump_val(inner_x, maxdispl ,x)
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)
    hs_size=hs.shape[2]-1

    hspdr = hs - hs[:, :, np.insert(np.arange(hs_size), 0, 0)]
    sm = 7 	# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = jump_val(inner_y,maxdispl ,y + 1)
    outer_x = jump_val(inner_x , maxdispl , x + 1)
    outer_r = minrad + r - 1

    return (outer_y, outer_x, outer_r)


#------------------------------------------------------------------------------
def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    N=len(angs)
    x_size=r.shape[0]
    y_size=r.shape[1]
    r_size=r.shape[2]
    y = np.zeros([N, x_size, y_size, r_size], dtype=int)
    x = np.zeros([N, x_size, y_size, r_size], dtype=int)
    for i in range(N):
        ang = angs[i]
        disp_angle1=np.cos(ang) * r
        disp_angle2=np.sin(ang) * r
        y[i, :, :, :] = np.round(y_0 - disp_angle1).astype(int)
        x[i, :, :, :] = np.round(x_0 + disp_angle2).astype(int)
    

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







