# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:15:00 2019

@author: cisguest
"""

import os
import glob
import sys
import random


# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 
  
# project imports
import datagenerators
import networks
import losses


sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen


def train(data_dir,
          atlas_file, 
          model,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          batch_size,
          load_model_file,
          data_loss,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    #atlas_vol = nib.load(atlas_file)
    #atlas_vol = atlas_vol.get_data()[np.newaxis, ..., np.newaxis]
    atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1] 
    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    #train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    train_vol_names = [#'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/otherNPZ/atlas_class_mapping.npz',
 #'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/otherNPZ/meanstats_T1_WARP.npz',
 #'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/otherNPZ/prob_atlas_41_class.npz',
# 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/otherNPZ/test_seg.npz',
 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/otherNPZ/test_vol.npz']
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else: # 'vm2double': 
        nf_enc = [f*2 for f in nf_enc]
        nf_dec = [f*2 for f in [32, 32, 32, 32, 32, 16, 16]]

    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC().loss        

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    gpu = '/gpu:%d' % 0 # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # prepare the model
        # in the CVPR layout, the model takes in [image_1, image_2] and outputs [warped_image_1, flow]
        # in the experiments, we use image_2 as atlas
        model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)

        # load initial weights
        if load_model_file is not None:
            print('loading', load_model_file)
            model.load_weights(load_model_file)

        # save first iteration
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # data generator
    nb_gpus = len(gpu_id.split(','))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'batch_size should be a multiple of the nr. of gpus. ' + \
        'Got batch_size %d, %d gpus' % (batch_size, nb_gpus)

    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size)
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    cvpr2018_gen = datagenerators.cvpr2018_gen(train_example_gen, atlas_vol_bs, batch_size=batch_size)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')

    # fit generator
    with tf.device(gpu):

        # multi-gpu support
        if nb_gpus > 1:
            save_callback = nrn_gen.ModelCheckpointParallel(save_file_name)
            mg_model = multi_gpu_model(model, gpus=nb_gpus)
        
        # single-gpu
        else:
            save_callback = ModelCheckpoint(save_file_name)
            mg_model = model

        # compile
        mg_model.compile(optimizer=Adam(lr=lr), 
                         loss=[data_loss, losses.Grad('l2').loss],
                         loss_weights=[1.0, reg_param])
            
        # fit
        mg_model.fit_generator(cvpr2018_gen, 
                               initial_epoch=initial_epoch,
                               epochs=nb_epochs,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch,
                               verbose=0)


data_dir = 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/test_data'
atlas_file = 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/data/atlas_norm.npz'
gpu_id = str(0)
model = 'vm2'
model_dir = 'C:/Users/cisguest/Downloads/voxelmorph-master/voxelmorph-master/my_models'
lr = 1e-4
epochs = 2
reg_param = 0.01
steps_per_epoch = 1 
batch_size = 1
load_model_file = None
data_loss = 'mse'

train(data_dir, atlas_file, model, model_dir, gpu_id, lr, epochs, reg_param, steps_per_epoch, batch_size, load_model_file, data_loss, initial_epoch=0)
