#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:38:47 2019

@author: mohammed
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:53:05 2018
@author: ilias
""" 
import cv2 
from os import listdir
from os.path import isfile, join
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
import numpy as np
subfolders = [f.path for f in os.scandir("/home/mohammed/DeepLearning/BalancedDataset/test") if f.is_dir() ] 
for action_folder in subfolders :
    
    action=os.path.basename(os.path.normpath(action_folder))
    if (action=="1") :
        

        os.mkdir("/home/mohammed/DeepLearning/patches_224/test/"+action)
        for video in os.listdir(action_folder):
            if (action=="1") :
               im = cv2.imread(action_folder+"/"+video)
        
            
            #im=cv2.resize(im, (224, 224))
               patches = image.extract_patches_2d(im, (224, 224),max_patches=1)
               print(patches.shape)
               count = 0

               for patche in patches :

                
                cv2.imwrite("/home/mohammed/DeepLearning/patches_224/test/%s/%s_patche%d.jpg"%(action, video , count) , patche)     # save frame as JPEG file
                count+=1

            
        
                 
         
        
        
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test1)     # save frame as JPEG file
#            count = 1
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test2)     # save frame as JPEG file
#            count = 2
#                    cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test3)     # save frame as JPEG file
#            count = 3
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , test4) 
#            count=4 # save frame as JPEG file
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , testc)     # save frame as JPEG file
#                 
#                 
#                 
#            count = 5
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest1)     # save frame as JPEG file
#            count = 6
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest2)     # save frame as JPEG file
#            count = 7
#                 
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest3)     # save frame as JPEG file
#            count = 8
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftest4) 
#            count=9 # save frame as JPEG file
#            cv2.imwrite("/home/mohammed/DeepLearning/dataset_aug/train/%s/%s_frame%d.jpg"%(action, video , count) , ftestc)     # save frame as JPEG file
#            
#            print ('Read a new frame: ', count)
#         