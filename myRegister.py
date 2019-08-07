# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:52:20 2019

@author: cisguest
"""


import sys
import numpy as np
import networks
sys.path.append('../ext/neuron')
import matplotlib.pyplot as plt




def mapImages(Im1, Im2, net, plotFlag = False):
   
    movingPath = Im1
    fixedPath = Im2
    nameMov = Im1[Im1.rfind('/')+1:Im1.find('.')]
    nameFix = Im2[Im2.rfind('/')+1:Im2.find('.')]
    titleX = nameMov + nameFix + 'X'
    titleY = nameMov + nameFix + 'Y'
    movIm = plt.imread(movingPath)
    movIm = movIm[:,:,0]
    #movIm=np.flip(movIm,0)  ####changed 
    #movIm=(np.roll(movIm,1)) ####changed 
    movIm = movIm.reshape(1,480,640,1)
    fixIm = plt.imread(fixedPath)
    fixIm = fixIm[:,:,0]
#    fixIm=np.flip(fixIm,0)  ####changed 
    fixIm = fixIm.reshape(1,480,640,1)  
    [moved, warp] = net.predict([movIm, fixIm])
    moved = moved.reshape(480,640)
    warp = warp.reshape(240,320,4)
    warpRegX = warp[:,:,1]
    warpRegY = warp[:,:,0]
    if plotFlag:
        plt.figure(figsize=(5,3),dpi=250)
        plt.imshow(movIm.reshape(480,640), cmap="gray")
        plt.title(nameMov)
        plt.figure(figsize=(5,3),dpi=250)
        plt.imshow(fixIm.reshape(480,640), cmap="gray")
        plt.title(nameFix)
        plt.figure(figsize=(5,3),dpi=250)
        plt.imshow(moved, cmap="gray")
        plt.title('Registered')
        plt.figure(figsize=(5,3),dpi=250)
        plt.imshow(warpRegX, cmap="gray")
        plt.title("X Registration")
        plt.colorbar()
        plt.figure(figsize=(5,3),dpi=250)
      
        plt.imshow(warpRegY, cmap="gray")
        plt.title('Y')
        plt.colorbar()
    return warpRegX, warpRegY, titleX, titleY
def getnet():
    model_file = 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/my_models/1500.h5'
    nf_enc = [16,32,32,32]
    nf_dec = [32,32,32,32,16,3]
#    atlas_vol = np.load(fixed)['vol'][np.newaxis, ..., np.newaxis]
#    vol_size = atlas_vol.shape[1:-1]
    vol_size = (480,640)
    bidir = 0 
    net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, bidir=bidir)
    net.load_weights(model_file)
    return net

#def main():
#    net = getnet()
#    mapImages('C:/Users/cisguest/Downloads/iris/dark iris/301.png','C:/Users/cisguest/Downloads/iris/dark iris/301.png',net)
#    
#if __name__ == "__main__":
#    main()
#%%
#[moved, warp] = net.predict([movIm, fixIm])
#plt.figure(figsize=(10,5),dpi=250)
#plt.subplot(221)
#plt.imshow(warp[0,:,:,0])
#plt.colorbar()
#plt.subplot(222)
#plt.imshow(warp[0,:,:,1])
#plt.colorbar()
#plt.subplot(223)
#plt.imshow(warp[0,:,:,2])
#plt.colorbar()
#plt.subplot(224)
#plt.imshow(warp[0,:,:,3])
#plt.colorbar()
#
##%%
#
#plt.figure(figsize=(10,5),dpi=250)
#plt.subplot(211)
#plt.imshow(np.sqrt(((warp[0,:,:,0])**2+(warp[0,:,:,1])**2)))
#plt.colorbar()
#plt.title('Mag')
#plt.subplot(212)
#plt.imshow(np.arctan2(warp[0,:,:,0],warp[0,:,:,1]))
#plt.colorbar()
#plt.title('Phase')