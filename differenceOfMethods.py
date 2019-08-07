# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:17:32 2019

@author: cisguest
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

with open('C:/Users/cisguest/Downloads/image pkl/points_before_and_after_matching_with_frame_number_320_399.pkl', 'rb') as pickle_file:
    contents = pickle.load(pickle_file)
    
image_before = np.array(contents['image_before']).reshape(80)[24:33]
image_after = np.array(contents['image_after']).reshape(80)[24:33]
initial_x_all = np.array(contents['initial_x_all']).reshape(80)[24:33]
initial_y_all = np.array(contents['initial_y_all']).reshape(80)[24:33]
next_x_all = np.array(contents['next_x_all']).reshape(80)[24:33]
next_y_all = np.array(contents['next_y_all']).reshape(80)[24:33]



#def histPlotter(arr1, arr2):
#    plt.hist(arr1)
#    plt.hist(arr2)
#    
#def Xhist():   
#    for i in range(4,13):
#        movement = np.array(next_x_all[i]) - np.array(initial_x_all[i])
##        #print(movement)
##        #movement = movement/255
##        maxVal = np.max(movement)
##        minVal = np.min(movement)
##        print('For pair', image_before[i], ' and ', image_after[i], maxVal, '----------', minVal)
##        plt.figure()
##        plt.hist(movement.ravel(), bins=1)
#        return movement.ravel()
#   
#    
#
#
#def YHist():
##    for i in range(4,13):
#        movement = np.array(next_y_all[i]) - np.array(initial_y_all[i])
##        #movement = movement/255
##        maxVal = np.max(movement)
##        minVal = np.min(movement)
##        print('For pair', image_before[i], ' and ', image_after[i], maxVal, '----------', minVal)
##        plt.figure()
##        plt.hist(movement.ravel(), bins=1)
#        return movement.ravel()
    
maskedImageNames = os.listdir('C:/Users/cisguest/Downloads/saccadeMasked')

xNames = []
yNames = []
for i in maskedImageNames:
    if i.find('X') != -1:
        xNames.append(i)
    else:
        yNames.append(i)
        
        
myHistPath = 'C:/Users/cisguest/Downloads/saccadeMasked/'
     
for i in range(0,9):
    firstHist = np.load(myHistPath+xNames[i])
    secondHist = np.array(next_x_all[i]) - np.array(initial_x_all[i])
    plt.figure()
    plt.subplot(121)
    plt.hist(firstHist[firstHist!=0].ravel())
    plt.xlim([-2,5])
    plt.title(xNames[i])
    plt.subplot(122)
    plt.hist(secondHist.ravel(), bins=500)
    plt.xlim([-2,5])
    plt.title(image_before[i])
    suffix, _ = xNames[i].split('.') 
    plt.savefig('C:/Users/cisguest/Downloads/plottedMethods/' + suffix +'.png')
    
for i in range(0,9):
    firstHist = np.load(myHistPath+yNames[i])
    secondHist = np.array(next_y_all[i]) - np.array(initial_y_all[i])
    plt.figure()
    plt.subplot(121)
    plt.hist(firstHist[firstHist!=0].ravel())
    plt.xlim([-2,5])
    plt.title(yNames[i])
    plt.subplot(122)
    plt.hist(secondHist.ravel(), bins=500)
    plt.xlim([-2,5])
    plt.title(image_before[i])
    suffix, _ = yNames[i].split('.')
    plt.savefig('C:/Users/cisguest/Downloads/plottedMethods/' + suffix +'.png')