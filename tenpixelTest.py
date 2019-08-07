# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:13:30 2019

@author: cisguest
"""

import numpy as np
import myRegister as mr
import matplotlib.pyplot as plt

im0 = 'C:/Users/cisguest/Downloads/iris/dark iris/301.png'

im10 = 'rolledByten.png'

im1 = 'rolledByOne.png'

im2 = 'rolledByTwo.png'

net = mr.getnet()

warpX0, warpY, titleX, titleY = mr.mapImages(im0, im0, net)
warpX1, warpY, titleX, titleY = mr.mapImages(im0, im1, net)
warpX2, warpY, titleX, titleY = mr.mapImages(im0, im2, net)
warpX10, warpY, titleX, titleY = mr.mapImages(im0, im10, net)


#plt.figure(figsize=(50,50))
#plt.imshow(warpX, cmap="gray")
#plt.xticks(np.arange(0,320,5))

print(warpX)
plt.figure()
plt.hist(warpX0.ravel())
plt.figure()
plt.hist(warpX1.ravel())
plt.figure()
plt.hist(warpX2.ravel())
plt.figure()
plt.hist(warpX10.ravel())

plt.figure()
plt.imshow(warpX0)
plt.figure()
plt.imshow(warpX1)
plt.figure()
plt.imshow(warpX2)
plt.figure()
plt.imshow(warpX10)
