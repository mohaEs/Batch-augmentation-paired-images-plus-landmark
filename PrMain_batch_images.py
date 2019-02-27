# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:58:55 2019

@author: Moha-Thinkpad
"""

## code for augmenting image + landmark locatios
# based on skimage
# and imgaug https://github.com/aleju/imgaug


from skimage import io
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import os
import imgaug as ia
from imgaug import augmenters as iaa
import glob
import os
from scipy import misc

# image source directory
SourcePath='./Images'
SourcePathCSV='./Landmarks'
SourcePathSeg='./paired Images'

# image destination directory
write_to_dir = "./augmented"

try:
    os.mkdir(write_to_dir)
except:
    print('destination folder is already exist')



# set your augmentation sequqnces here
# in a list called AugCongigList    
    
    
AugCongigList=[ 
        iaa.Sequential([iaa.Fliplr(1, name="Flipper")
                      ], name='first config, just flip')
        ,              
        iaa.Sequential([iaa.Fliplr(1, name="Flipper"),
                      iaa.Affine(scale={"x": 0.8, "y": 0.9},  
                                       translate_percent={"x": 0.2, "y":  0.1},   
                                       rotate= 45, name='affine 1')] , name='second config, sequential, flip + affine')
        ]    



for filename in glob.glob(SourcePath+'/*.png'): #assuming png
    
    FileName=filename.replace(SourcePath,'')
    FileName=FileName[:len(FileName)-4]
    
    
    Image = io.imread(filename)    
    Image_seg=io.imread(SourcePathSeg+FileName+'.png')
    
    Landmarks = genfromtxt(SourcePathCSV+FileName+'.csv', delimiter=',')    
    Landmarks = Landmarks.astype(int)
    Landmarks = Landmarks[1:] # remove the first row because it is just axis label    
    
    #### visualization
#    plt.figure()
#    plt.imshow(Image)
#    plt.plot(Landmarks[0,1],Landmarks[0,0],marker="s",color='red')
#    plt.plot(Landmarks[1,1],Landmarks[1,0],marker="s",color='red')
#    plt.plot(Landmarks[2,1],Landmarks[2,0],marker="s",color='red')
#    plt.plot(Landmarks[3,1],Landmarks[3,0],marker="s",color='red')
#    plt.plot(Landmarks[4,1],Landmarks[4,0],marker="s",color='red')
    # The augmenters expect a list of imgaug.KeypointsOnImage.
    try:
        images=np.zeros(shape=[1,Image.shape[0],Image.shape[1],Image.shape[2]], dtype='uint8')
        images[0,:,:,:]=Image
        images_seg=np.zeros(shape=[1,Image.shape[0],Image.shape[1],Image.shape[2]], dtype='uint8')
        images_seg[0,:,:,:]=Image_seg
    except:
        images=np.zeros(shape=[1,Image.shape[0],Image.shape[1]], dtype='uint8')
        images[0,:,:]=Image
        images_seg=np.zeros(shape=[1,Image.shape[0],Image.shape[1]], dtype='uint8')
        images_seg[0,:,:]=Image_seg
        
    # Generate random keypoints.
    # The augmenters expect a list of imgaug.KeypointsOnImage.
    keypoints_on_images = []
    for image in images:
        keypoints = []
        for _ in range(len(Landmarks)):
            keypoints.append(ia.Keypoint(x=Landmarks[_,1], y=Landmarks[_,0]))
        keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))
            

    for ConfCounter in range(len(AugCongigList)):
    
        seq=AugCongigList[ConfCounter]
    
        seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
        
        # augment keypoints and images
        images_aug = seq_det.augment_images(images)
        images_aug_seg = seq_det.augment_images(images_seg)
        transformed_keypoints = seq_det.augment_keypoints(keypoints_on_images)
        
        X_new=[]
        Y_new=[]
        # Example code to show each image and print the new keypoints coordinates
        for  keypoints_after in transformed_keypoints:
            for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
                x_new, y_new = keypoint.x, keypoint.y
                X_new.append(x_new)
                Y_new.append(y_new)
        
        newLandmarks=np.zeros(Landmarks.shape) 
        newLandmarks[:,0]=np.asarray(Y_new)
        newLandmarks[:,1]=np.asarray(X_new)
        newLandmarks=newLandmarks.astype(int)

#        plt.figure()
#        plt.imshow(images_aug[0,:,:])
#        plt.plot(newLandmarks[0,1],newLandmarks[0,0],marker="s",color='red')
#        plt.plot(newLandmarks[1,1],newLandmarks[1,0],marker="s",color='red')
#        plt.plot(newLandmarks[2,1],newLandmarks[2,0],marker="s",color='red')
#        plt.plot(newLandmarks[3,1],newLandmarks[3,0],marker="s",color='red')
#        plt.plot(newLandmarks[4,1],newLandmarks[4,0],marker="s",color='red')
    
        try:
            misc.imsave(write_to_dir+FileName+'_'+str(ConfCounter)+'_aug.png', images_aug[0,:,:,:])
            misc.imsave(write_to_dir+FileName+'_'+str(ConfCounter)+'_pair_aug.png', images_aug_seg[0,:,:,:])
        except:
            misc.imsave(write_to_dir+FileName+'_'+str(ConfCounter)+'_aug.png', images_aug[0,:,:])
            misc.imsave(write_to_dir+FileName+'_'+str(ConfCounter)+'_pair_aug.png', images_aug_seg[0,:,:])


        np.savetxt(write_to_dir+FileName+'_'+str(ConfCounter)+'_aug.csv', 
                   newLandmarks , delimiter=",", fmt='%i' , header='row,col')
                
        text_file = open(write_to_dir+FileName+'_'+str(ConfCounter)+'_info.txt', "w")
        text_file.write("Augmentation Info " + '\n' + 'name:' + seq.name + '\n' +'\%s' % seq)
        text_file.close()