# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:04:38 2019

@author: cisguest

delete all 301X.npy files from savepath before you run this
"""


import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

def getaxis(imName):
    pos = imName.rfind('.')
    axis = pos-1
    return imName[axis]

def getMaskName(imName):
    firstImName = imName[0:3]
    axisName = getaxis(imName)
    maskPath = 'C:/Users/cisguest/Downloads/masks/new_mask/saccades/'
    maskName = maskPath + firstImName + '.png'
    return maskName, firstImName +axisName

def getMask(imName):
    maskName, saveName = getMaskName(imName)
    mask = Image.open(maskName)
    mask = mask.resize((320,240))
    mask = np.array(mask)
    return mask, saveName

def applyMask(image, mask):
    before = image
    after = before * mask
    after = after/255
    return after

def checkNPY(name):
    _, exten = name.split('.')
    if exten == 'npy':
        return True
    return False

def plotHist(array, saveName):
    array = array[array!=0]
    plt.figure() 
    plt.hist(array.ravel())
    print(np.histogram(array.ravel()))
    mean = '\u03BC=' + str(np.mean(array[array!=0]))
    sd = '\u03C3=' + str(np.std(array[array!=0]))
    title = mean + 'for '+ saveName + sd
    plt.title(title)
#    array = array[array != 0]
#    print(saveName, ' median = ', np.median(array), ' mean = ', np.mean(array))
    return array    

def runOperation():
     allImageNames = os.listdir('C:/Users/cisguest/Downloads/RMask/saccades')
     savePath = 'C:/Users/cisguest/Downloads/RMask/saccades/'
     for i in range(len(allImageNames)):
         currentName = 'C:/Users/cisguest/Downloads/RMask/saccades/' + allImageNames[i]
         if(checkNPY(currentName)):
             currentImage = np.load(currentName)
             mask, saveName = getMask(allImageNames[i])
    #         print("Mask", mask.shape)
    #         print(currentName, "shape", currentImage.shape)
             maskedImage = applyMask(currentImage, mask)
             arr = plotHist(maskedImage, saveName)
#             print(maskedImage, maskedImage.shape, np.max(maskedImage), np.min(maskedImage))
             np.save(savePath+saveName, maskedImage)
#             return arr
         
def main():
    runOperation()
if __name__ == "__main__":
    main()