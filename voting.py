#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:34:14 2019

@author: mohammed
"""

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from pyimagesearch import config
from imutils import paths
import numpy as np
import os
from keras.preprocessing import image
from sklearn.feature_extraction import image
import keras

from keras.models import load_model
from collections import Counter
from skimage.io import imread_collection
import cv2
from keras.utils.vis_utils import plot_model

#your path 
col_dir = 'BalancedDataset/test/0/*.png'
patcheslist =  list()
patchenums=[20,50,100,150,200]
classes =  list()
print("je suis",len(col_dir))
#creating a collection with the available images
imgs = imread_collection(col_dir)
path="saved-model-48-0.76.hdf5"
model = load_model(path)
#img = image.load_img('test.jpg', target_size=(48, 48))

for patchenum in patchenums:
    classes.clear()
    for img in imgs :
        img=cv2.resize(img,(460,320))
        patches = image.extract_patches_2d(img, (48, 48),max_patches=patchenum)
        patcheslist.clear()
        for patche in patches :

           img = np.expand_dims(patche, axis=0)
           img = img.astype('float32')
           img /= 255
           classe = model.predict_classes(img)
           patcheslist.append(classe[0])
        
        if((patcheslist.count(0)/len(patcheslist))>0.5):
            classes.append(0)
        else : 
            classes.append(1)


    if(len(classes)!=0):
         print(classes.count(0)/len(classes))

    
 