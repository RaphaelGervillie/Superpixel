
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2

from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.segmentation import felzenszwalb, slic, mark_boundaries
from skimage.util import img_as_float
from time import time

from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from resizeimage import resizeimage



def sp_idx(s, index = True):
    u = np.unique(s)
    return [np.where(s == i) for i in u]





def EV(image,segments):


    """
    image : matrix of a color image
    segments : matrix of the superpixel segmentation

    Allows you to measure the color homogeneity of a superpixel.
    """

    R=image[:,:,0]
    G=image[:,:,1]
    B=image[:,:,2]
    
    
    sigma_image_R=np.sum(abs(R-np.mean(R))**2)
    mean_R=np.mean(R)
    regions_R = measure.regionprops(segments, intensity_image=R)
    sigma_segmentation_R=np.sum([r.area*(r.mean_intensity -mean_R)**2 for r in regions_R])
    R_met=sigma_segmentation_R/sigma_image_R
    
    sigma_image_G=np.sum(abs(G-np.mean(G))**2)
    mean_G=np.mean(G)
    regions_G = measure.regionprops(segments, intensity_image=G)
    sigma_segmentation_G=np.sum([r.area*(r.mean_intensity -mean_G)**2 for r in regions_G])
    G_met=sigma_segmentation_G/sigma_image_G
    
    sigma_image_B=np.sum(abs(B-np.mean(B))**2)
    mean_B=np.mean(B)
    regions_B = measure.regionprops(segments, intensity_image=B)
    sigma_segmentation_B=np.sum([r.area*(r.mean_intensity -mean_B)**2 for r in regions_B])
    B_met=sigma_segmentation_B/sigma_image_B
    return np.mean(np.array([R_met,G_met,B_met]))
    
    
def C(image,segments):

    """
    image : matrix of a color image
    segments : matrix of the superpixel segmentation

    Allows you to measure the compactness of a superpixel, individually.
    """

    img=image[:,:,0]
    regions = measure.regionprops(segments, intensity_image=img)
    compactness=np.mean([(np.pi*4*r.area)/r.perimeter**2 for r in regions])
    return compactness
    
def SRC(image,segments):

    """
    image : matrix of a color image
    segments : matrix of the superpixel segmentation

    Used to measure the compactness of superpixels.
    This metric will perform well for square shapes, circles or hexagons, so we try to maximize this criterion.
    """
    
    R=image[:,:,0]
    regions_R = measure.regionprops(segments, intensity_image=R)
    
    superpixel_list = sp_idx(segments)
    superpixel      = [idx for idx in superpixel_list]

    perim_sk=np.array([r.perimeter for r in regions_R])
    area_sk=np.array([r.area for r in regions_R])
    area_hull=np.array([r.convex_area for r in regions_R])

    perim_hull=np.zeros((len(superpixel),))
    for i in range(len(superpixel)):
        XY=np.array([superpixel[i][0],superpixel[i][1]]).T
        hull = ConvexHull(XY)
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        perimeter = np.sum([euclidean(x, y) for x, y in zip(XY[vertices], XY[vertices][1:])])
        perim_hull[i]=perimeter

    perim_hull = perim_hull[1:]

    CRS=(perim_hull/area_hull)/(perim_sk/area_sk)


    VXY_ls=np.zeros((len(superpixel),))
    for i in range(len(superpixel)):
        sig_X = (np.std(superpixel[i][0]))
        sig_Y = (np.std(superpixel[i][1]))
        VXY= min(sig_X,sig_Y)/max(sig_X,sig_Y)
        VXY_ls[i]=VXY

    VXY_ls = VXY_ls[1:]

    N=image.shape[0]*image.shape[1]
    SRC=np.sum((area_sk/N)*CRS*VXY_ls)
    return SRC





def Undersegmentation_error(segments_truth, segments):

    """
    segments_truth : matrix ground truth 
    segments_slic : matrix of segments (SLIC for example)

    To calculate the proximity level of superpixels to image borders

    """
    
    S_truth=np.unique(segments_truth.flatten())
    results=[]

    for gt in S_truth:
        St0 = segments_truth==gt
        UE=0
        for i in np.unique(segments[St0]): 
            in_border=(segments[St0]==i).sum() 
            out_border=(segments==i).sum() - in_border
            UE += min(in_border,out_border)

        results.append(UE) 
    N=segments_truth.shape[0]*segments_truth.shape[1]
        
    return np.sum(results)/ N # normalise 




def EV1(image,segments):
    R=image[:,:,0]
    G=image[:,:,1]
    B=image[:,:,2]
    
    
    regions_R = measure.regionprops(segments, intensity_image=R)
    mean_R=[r.mean_intensity for r in regions_R]
    
    regions_G = measure.regionprops(segments, intensity_image=G)
    mean_G=[r.mean_intensity for r in regions_G]
    
    regions_B = measure.regionprops(segments, intensity_image=B)
    mean_B=[r.mean_intensity for r in regions_B]
    
    z=np.array([mean_R,mean_G,mean_B])
    return z
    


def EV2(image,segments):
    R=image[:,:,0]
    G=image[:,:,1]
    B=image[:,:,2]
    mean_R=np.mean(R)
    mean_G=np.mean(G)
    mean_B=np.mean(B)
    


    regions_R = measure.regionprops(segments, intensity_image=R)
    sigma_segmentation_R=[(r.mean_intensity -mean_R)**2 for r in regions_R]
    

    regions_G = measure.regionprops(segments, intensity_image=G)
    sigma_segmentation_G=[(r.mean_intensity -mean_G)**2 for r in regions_G]

    

    regions_B = measure.regionprops(segments, intensity_image=B)
    sigma_segmentation_B=[(r.mean_intensity -mean_B)**2 for r in regions_B]
    z=np.array([sigma_segmentation_B,sigma_segmentation_G,sigma_segmentation_R])

    return np.mean(z,axis=0)



def Image_colorfull_var(image,segments):

    """
    image : matrix image
    segments : matrix of the superpixel segmentation

    Allows you to observe the variation of the color on each superpixel.
    """

    vis = np.zeros(image.shape[:2], dtype="float")
    test=EV2(image,segments)


    for v in np.unique(segments-1):
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0

        vis[segments == v] = test[v-1]

    vis = rescale_intensity(vis, out_range=(0, 1)).astype("float")

    plt.imshow(vis,plt.get_cmap('gray'))
    plt.axis('off')
    plt.savefig('std_superpixel')


def Image_colorfull_mean(image,segments):
    """
    image : matrix image
    segments : matrix of the superpixel segmentation

   
    Allows you to observe the average color on each superpixel.

    """

    test=EV1(image,segments)
    vis = np.zeros(image.shape,dtype='float')
    for v in np.unique(segments-1):
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0

        vis[:,:,0][segments == v] = test[0][v-1]
        vis[:,:,1][segments == v] = test[1][v-1]
        vis[:,:,2][segments == v] = test[2][v-1]

    plt.imshow(vis.astype(int))
    plt.axis('off')
    plt.savefig('mean_superpixel')	



def test_slic_params(K,M,final_ls):
    """
    K :  list or Integer
    M : list or Integer
    final_ls : list 

    This function takes as input data the list of indexes allowing access to your
    ground truth and your images.

    To use this function, it is advisable to set one parameter and vary the other.
    For example we can use it like this: test_slic_params (k = 100, M = [1,10,100], final_ls)
    In this example, the parameter K of the SLIC algorithm is fixed at 100 and the compactness parameter is varied.

    For each parameter combination, it returns the average result of the SRC, EV and UE metrics calculated on your dataset
    """
    
    results_dic={}
    
    if type(K)==list and type(M)==int :
        for i,k in enumerate(K):
            print("k = {}".format(k))
            results=np.zeros((3,len(final_ls)))
            for idx,path in enumerate(final_ls):   
                
                image = cv2.imread('Data/train/{}.jpg'.format(path))
                GT =  np.load("Data/ground_truth/{}.npy".format(path))
                segments = slic(img_as_float(image), n_segments = k, compactness=M)
      
                
                results[0][idx]=SRC(image,segments)
                results[1][idx]=EV(image,segments)
                results[2][idx]=Undersegmentation_error(GT,segments)

                if (idx%20)==0:
                    print('Image : {}, SRC : {}, EV : {}, UE : {}'.format(
                        idx,results[0][idx],results[1][idx],results[2][idx]))
            results_dic[k]=results
            print('-'*10)
            
        result_2=np.zeros((len(K),3))
        for idx,k in enumerate(K):
            result_2[idx,0]= results_dic[k][0].mean()
            result_2[idx,1]= results_dic[k][1].mean()
            result_2[idx,2]= results_dic[k][2].mean()
            
        df=pd.DataFrame(result_2,index=K,columns=["SRC","EV","UE"])
    
        return df
    
    if type(K)==int and type(M)==list :
        for i,m in enumerate(M):
            print("m = {}".format(m))
            results=np.zeros((3,len(final_ls)))
            for idx,path in enumerate(final_ls):   
                
                image = cv2.imread('Data/train/{}.jpg'.format(path))
                GT =  np.load("Data/ground_truth/{}.npy".format(path))
                segments = slic(img_as_float(image), n_segments = K,compactness=m)
                
                results[0][idx]=SRC(image,segments)
                results[1][idx]=EV(image,segments)
                results[2][idx]=Undersegmentation_error(GT,segments)

                if (idx%20)==0:
                    print('Image : {}, SRC : {}, EV : {}, UE : {}'.format(
                        idx,results[0][idx],results[1][idx],results[2][idx]))
            results_dic[m]=results
            print('-'*10)
            
        result_2=np.zeros((len(M),3))
        for idx,m in enumerate(M):
            result_2[idx,0]= results_dic[m][0].mean()
            result_2[idx,1]= results_dic[m][1].mean()
            result_2[idx,2]= results_dic[m][2].mean()
            
        df=pd.DataFrame(result_2,index=M,columns=["SRC","EV","UE"])
        return df



def test_FH_params(scale,final_ls):

    """
    scale : list
    final_ls : list 

    This function takes as input data the list of indexes allowing access to your
    ground truth and your images.

    It also takes the scale parameter as a list.
    This allows to calculate the average scrore on all your images of the
    SRC, EV and UE metrics.

    """

    results_dic={}

    for i,z in enumerate(scale):
        print("z = {}".format(z))
        results=np.zeros((3,len(final_ls)))
        for idx,path in enumerate(final_ls):   

            image = cv2.imread('Data/train/{}.jpg'.format(path))
            GT =  np.load("Data/ground_truth/{}.npy".format(path))
            segments = felzenszwalb(img_as_float(image), scale = z)
            try : 
                results[0][idx]=SRC(image,segments)
            except:
                results[0][idx]= 0
            results[1][idx]=EV(image,segments)
            results[2][idx]=Undersegmentation_error(GT,segments)

            if (idx%20)==0:
                print('Image : {}, SRC : {}, EV : {}, UE : {}'.format(
                    idx,results[0][idx],results[1][idx],results[2][idx]))
        results_dic[z]=results
        print('-'*10)

    result_2=np.zeros((len(scale),3))
    for idx,z in enumerate(scale):
        result_2[idx,0]= results_dic[z][0].mean()
        result_2[idx,1]= results_dic[z][1].mean()
        result_2[idx,2]= results_dic[z][2].mean()

    df=pd.DataFrame(result_2,index=scale,columns=["SRC","EV","UE"])

    return df

def test_FH_time(scale,final_ls):

    """
    scale : list
    final_ls : list 

    This function takes as input data the list of indexes allowing access to your
    ground truth and your images. It also takes the scale parameter as a list.
    This allows to calculate the average time taken by the algorithm of Felznwalb and Huttenlocher
    to segment according to the parameter z.

    """

    results_dic={}

    for i,z in enumerate(scale):
        print("z = {}".format(z))
        results=np.zeros((1,len(final_ls)))
        for idx,path in enumerate(final_ls):   

            image = cv2.imread('Data/train/{}.jpg'.format(path))
            GT =  np.load("Data/ground_truth/{}.npy".format(path))
            t0 = time.time()
            segments = felzenszwalb(img_as_float(image), scale = z)
            tend = time.time()-t0
            results[0][idx]=tend

            if (idx%20)==0:
                print('Image : {}, Time : {}'.format(idx,results[0][idx]))
        results_dic[z]=results
        print('-'*10)

    result_2=np.zeros((len(scale),1))
    for idx,z in enumerate(scale):
        result_2[idx,0]= results_dic[z][0].mean()
    df=pd.DataFrame(result_2,index=scale,columns=["Time"])

    return df


def Open_seg(path,width=481,length=321):
    """Open_seg(path,width,length)** :
    path : path to .seg file
    width : integer
    length : integer

    This function takes the length and width of the image as input.
    All of the images I used were the same length and width because they were taken from Berkley's dataset.
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    If you use this function for berkley's dataset all images are normally 481 rows and 321 columns format
    so you don't have to specify this parameter.
    """
    z=[]
    with open(path,'r') as f:
        for line in f:
            for word in line.split("data"):
                z.append(word)

    ok=z[12:]
    df=pd.DataFrame([i.split(' ') for i in ok])
    df.iloc[:,3]=df.iloc[:,3].apply(lambda x : x.split('\n')[0])
    df=df.astype(int)

    arr=np.array(df)
    z=np.zeros((width,length))
    for i,j in enumerate(arr):
        z[j[1],j[2]:j[3]+1]=j[0]
    z=z.astype("uint16")
    return z