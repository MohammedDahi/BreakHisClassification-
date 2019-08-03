#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:39:44 2019

@author: mohammed
"""
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from pyimagesearch import config
from imutils import paths
import numpy as np
import os
from keras.preprocessing import image

size=48
BS=32
totalTest = len(list(paths.list_images("/home/mohammed/DeepLearning/patches/test")))
from keras.models import load_model
path="patch2d"
models= os.listdir(path)



for model in models : 
    print(model)
    model = load_model(path+"/"+model)
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    
    
    testGen = valAug.flow_from_directory(
        "/home/mohammed/DeepLearning/patches/test",
        class_mode="categorical",
        target_size=(size, size),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)
    print("[INFO] evaluating network...")
    testGen.reset()
    predIdxs = model.predict_generator(testGen,
        steps=(totalTest // BS) + 1)
    
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testGen.classes, predIdxs,
        target_names=testGen.class_indices.keys()))
    
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testGen.classes, predIdxs)
    total = sum(sum(cm))
    print(total)
    print(cm[0,0])
    acc = (cm[0,0] + cm[1,1]) / total
    sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
    specificity = cm[1,1] / (cm[1,0] + cm[1,1])
    
    
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print(acc)
    print(sensitivity)
    print(specificity)


   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    