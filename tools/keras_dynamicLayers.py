#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:42:47 2020

@author: sam
"""
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle
import keras
from keras import layers
from keras import backend as K
#
class trainableInput(keras.layers.Layer):
    #layer that returns it's weight on call
    #(pretty much gives you an easily manipulable trainable variable)
    def __init__(self, p_shape=(100,),initializer='zeros',**kwargs):
        self.p_shape = p_shape
        self.initializer = initializer
        super(trainableInput, self).__init__(**kwargs)
    def build(self,input_shape):
        self._P = self.add_weight(name='P',
                    shape=self.p_shape,
                    initializer=self.initializer,#keras.initializers.glorot_uniform(),
                    trainable=True,)
        super(trainableInput, self).build(input_shape)
    def get_config(self):
        config = {'p_shape': self.p_shape,
                  'initializer':self.initializer,
                }
        base_config = super(trainableInput, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def call(self,inputs):
        out = [-1,]
        out.extend(list(self.p_shape))
        n_tot = 1
        for i in self._P.shape.as_list():
            n_tot *= i
        dummy = inputs
        n_in = dummy.shape.as_list()[1]
        if n_tot>n_in:
            dummy = K.tile(dummy,n_tot//n_in+1)
        dummy = dummy[:,:n_tot]
        dummy = K.reshape(dummy,out)
        dummy = dummy * 0 + self._P
        return dummy

    def compute_output_shape(self, input_shape):
        out = [None,]
        out.extend(list(self.p_shape))
        return tuple(out)


class ScalingLayer(keras.layers.Layer):
    #applies a linear scaling (no bias) to all elements in a tensor
    def __init__(self,init_val=1, **kwargs):
        self.init_val = init_val
        super(ScalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._s = self.add_weight(name='s',
                                  shape=(1,),
                                  initializer=keras.initializers.constant(self.init_val),
                                  trainable=True,)
        super(ScalingLayer, self).build(input_shape)

    def call(self, inputs):
        return self._s * inputs

class BiasLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(1,),
                                    initializer='zeros',
                                    trainable=True)
        super(BiasLayer,self).build(input_shape)
    def call(self, x):
        return x + self.bias

class initial_condition_layer(keras.layers.Layer):

    def __init__(self, n=100, n_genes=3, G=None, init_zeros=False, noise=False,
                 gene_specific=False,**kwargs):
        self.n = n
        #check if got passed as list, turn into np arrray if so:
        if isinstance(G,list):
            G=np.array(G)
        #Genetic Mask matrix
        if G is None:
            G = np.zeros((n,n_genes))
            split = np.linspace(0,n,n_genes+1).astype(int)
            for i in range(n_genes):
                st = split[i]
                en = split[i+1]
                G[st:en,i] = 1
        #If ID is passed in a non-one-hot array
        elif len(G.shape)==1:
            G0 = G.copy()
            G = np.zeros((n,n_genes))
            for i,v in enumerate(G0):
                #if motor neuron
                if v==-1:
                    G[i]=1
                #if neural type
                else:
                    G[i,v] = 1
        self.G = tf.constant(G,dtype='float32')
        #Whether this is a layer that just returns a zero tensor or is trainable layer
        self.init_zeros = init_zeros
        #whether to add noise to Q0
        self.noise = noise
        #Whether each knockdown condition learns its own initialization
        self.gene_specific = gene_specific
        super(initial_condition_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Creates all trainable weight in mechanistic model
        self._Q0 = self.add_weight(name='Q0',
                                    shape=(self.n,),
                                    initializer='zeros',#pos_init,#keras.initializers.glorot_uniform(),
                                    trainable=True,)
        if self.gene_specific:
            self._Qgene = self.add_weight(name='Qgene',
                                          shape=(self.G.shape.as_list()[1],self.n),
                                          initializer='zeros',
                                          trainable=True,
                                          )
        super(initial_condition_layer, self).build(input_shape)

    def get_config(self):
        config = {'n': self.n,
                  'G': np.array(K.eval(self.G)),
                  'init_zeros': self.init_zeros,
                  'noise': self.noise,
                  'gene_specific': self.gene_specific,
                }
        base_config = super(initial_condition_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def call(self, inputs):
        gamma = inputs
        #conditional genetic matrix
        G = tf.einsum('ij,kj->kij',self.G, gamma)
        G = K.max(G,axis=-1)
        Q = self._Q0
        if self.gene_specific:
            gamma_inv = 1-gamma
            delta = tf.einsum('kj,ji->kij',gamma_inv, self._Qgene)
            delta = K.sum(delta,axis=-1)
            Q = Q + delta
        Q = tf.math.multiply(Q,G)
        if self.init_zeros:
            Q = 0*Q
        if self.noise:
            Q = Q + K.random_normal(tf.shape(Q),0,.03)
        return  Q#TODO: consider noise sampling

    def compute_output_shape(self, input_shape):
        return (None, self.n)

