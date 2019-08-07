# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:52:05 2019

@author: aa6692
"""
import numpy as np
import os
import matplotlib.pyplot as plt
imageNames = []
images = []
imageNames = os.listdir('C:/Users/cisguest/Downloads/iris/dark iris/')

for each_image in imageNames:
    name, _ = each_image.split('.')
    imagePath = 'C:/Users/cisguest/Downloads/iris/dark iris/' + each_image
    tempIm = plt.imread(imagePath)
    tempIm = tempIm[:,:,0]
    filename = 'C:/Users/cisguest/Downloads/iris/trainpz/'+ name + '.npz'
    np.savez(filename, vol_data=tempIm)
    
    