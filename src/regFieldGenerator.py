# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:58:46 2019

@author: cisguest
"""

import myRegister as mr
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imsave as imsave

def getAllImages(numImg = 10):
    allImages = []
    path = 'C:/Users/cisguest/Downloads/iris/dark iris/'
    extension = '.png'
    imageNames = np.arange(343,354,1, dtype=int)
    for i in range(numImg):        
        currentImageName = path+str(imageNames[i])+extension
        #currentImage = plt.imread(currentImageName)
        allImages.append(currentImageName)
    
    return np.asarray(allImages)

def getPairOfImages(allImages, flag):

    I1, I2 = allImages[flag], allImages[flag+1]
    return I1, I2

def runOperation():
    allImages = getAllImages()
    net = mr.getnet()
    for i in range(9):
        firstIm, secondIm = getPairOfImages(allImages, i)
        warpX, warpY, titleX, titleY = mr.mapImages(firstIm, secondIm, net)
        print(np.max(warpX), np.min(warpX))
#        print(warpX.shape)
#        print(warpY.shape)
#        print(type(warpX))
#        print(type(warpY))
        titleX = 'C:/Users/cisguest/Downloads/RMask/' + titleX #+'.png'
        titleY = 'C:/Users/cisguest/Downloads/RMask/' + titleY #+'.png'
#        print(titleX)
#        print(titleY)
        #warpX = np.rot90(warpX, k=2)
        #warpY = np.rot90(warpY, k=2)
#        plt.figure(figsize=(5,3),dpi=250)
#        plt.subplot(121)
#        plt.imshow(warpX, cmap="gray")
#        plt.colorbar()
#        plt.title(titleX)
#        plt.subplot(122)
#        plt.imshow(warpY, cmap="gray")
#        plt.colorbar()
#        plt.title(titleY)
#        imsave(titleX, warpX)
#        imsave(titleY, warpY)
        np.save(titleX, warpX)
        np.save(titleY, warpY)
                
def main():
    runOperation()
    
if __name__ == "__main__":
    main()