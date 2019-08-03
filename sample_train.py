#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 23:57:19 2019

@author: mohammed
"""
import numpy
import os
import random
patients_b = list()
patients_m = list()

cp = 0
for dirname, dirnames, filenames in os.walk('/home/mohammed/DeepLearning/DataSet/BreaKHis_v1/histology_slides/breast'):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        str = os.path.join(subdirname)
        if str.startswith("SOB_B") : 
            patients_b.append(os.path.join(subdirname))
        if str.startswith("SOB_M") : 
            patients_m.append(os.path.join(subdirname))    


print(cp)
train_patients_b=random.sample(patients_b, int(24*0.7))
train_patients_m=random.sample(patients_m, int(58*0.7))
train_patients=train_patients_b+train_patients_m

numpy.savetxt("sample.txt",train_patients,fmt='%s')

