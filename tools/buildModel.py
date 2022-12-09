#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 08:57:38 2022

@author: sam
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import matplotlib.pyplot as plt
config = tf.ConfigProto()
config.gpu_options.allow_growth = True#False
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()

def buildModel(folder,connectivity=None,
               n_neuron=20**2, n_peptide=2,
               noise=1e-4,
               syn_fractions=np.array([0,0,.05,.05,.33,.05])):
#    syn_fractions /= syn_fractions.sum()
    #folder='./analysisScripts/dualNet/bigRun/TEST/'
    
    if not os.path.isdir(folder):
            os.mkdir(folder)
    np.save(f'{folder}syn_fractions.npy',np.array(syn_fractions))
    
    
    #In[]
    '''Build the model'''
    ####################################################
    log_train = False
    reduce_training_variables=0#True
#    n_neuron=50**2
#    n_peptide=2
    sq=int(n_neuron**.5)
    init='glorot_uniform'
    trainable = [False for i in range(7)]
    trainable[6] = True
    trainable[2] = True
    trainable[4] = True
    trainable[5] = True
    magnitude_fit=True
    
    n_genes=1
    ablate=True
    constrain_measure=False
    sigmoid_measure=False
    connectivity_mask = True
    synapse_mask=True
    
    if connectivity is None:
        from .lattices import WS_network
        connectivity = WS_network(beta=1e-3,n_neuron=n_neuron,k=8,shuffle=True)
#    noise = 1e-4#'exp'
    ####################################################
    from .model import build_model
    model, latent_model, THETA = build_model(log_train,n_neuron,n_peptide,init,trainable,
                    magnitude_fit,n_genes,ablate,constrain_measure,
                    sigmoid_measure,connectivity_mask,synapse_mask,
                    noise)
    
    '''set parameters'''
    from .initialize_parameters import initialize_parameters
    initialize_parameters(model,THETA,n_neuron,n_peptide,connectivity,sigmoid_measure)
    
    return model,latent_model,THETA
    
    
    
    
    
    
    
    
    
    