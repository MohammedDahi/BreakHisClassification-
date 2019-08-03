#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:01:41 2019

@author: mohammed
"""
import os
import cv2
import random 
path = '/home/mohammed/DeepLearning/BalancedDataset/'
with open("allImages.txt") as f:
   liste = f.read().splitlines() 
filenames=list()
filenames_train=list()
patients=list()
patients_m=list()
patients_b=list()

for image_path in liste: 
    png=image_path.split("/")[13]
    filenames.append(png)
    
with open('sample.txt') as f:
    liste2 = f.read().splitlines()   
    
for patient in liste2: 
    p=patient.split("-")[1]
    if "_B_" in patient :
            patients_b.append(p)   
    if "_M_" in patient:
        patients_m.append(p)
        
patients_m=random.sample(patients_m,int(0.65*len(patients_m)))

patients= patients_m+patients_b
for p in patients :
      for f in filenames:
        if p  in f :
                 #print(f)
                 filenames_train.append(f)

train_benigns=list()    
train_malignants=list()         
test_benigns=list()    
test_malignants=list()         



filenames_test=[x for x in filenames if not x in filenames_train]          
      
for a in filenames_train:
    if "_B" in a :
        train_benigns.append(a)
    if "_M" in a :
        train_malignants.append(a)    

for a in filenames_test:
    if "_B" in a :
        test_benigns.append(a)
    if "_M" in a :
        test_malignants.append(a)    
        
print(len(train_benigns))
print(len(train_malignants))
print(len(test_benigns))
print(len(test_malignants))


print(len(filenames_train))
print(len(filenames_test))


   
for image_path in liste: 
    
    img = cv2.imread(image_path, 1)
    filename=image_path.split("/")[13]
    
    
    
    if filename in train_malignants : 
                save_path = os.path.join(path,"train/malignant/")
                print(os.path.join(save_path,filename))
                cv2.imwrite(os.path.join(save_path,filename), img)
    if filename in train_benigns : 
            
                save_path = os.path.join(path,"train/benign/")
       
                print(os.path.join(save_path,filename))

                cv2.imwrite(os.path.join(save_path,filename), img)
    if filename in test_malignants : 
                save_path = os.path.join(path,"test/malignant/")
                print(os.path.join(save_path,filename))
                cv2.imwrite(os.path.join(save_path,filename), img)

    if filename in test_benigns : 

                save_path = os.path.join(path,"test/benign/")
       
                print(os.path.join(save_path,filename))

                cv2.imwrite(os.path.join(save_path,filename), img)
