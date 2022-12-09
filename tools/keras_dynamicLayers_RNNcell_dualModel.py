#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:08:47 2021

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
from tensorflow.python.training.tracking.data_structures import NoDependency
from keras.layers import InputSpec
import numpy as np

class rnnCell_dualNet_(keras.layers.Layer):
    """
    This class defines a recurrent neural network (RNN) cell for use in the Keras deep learning framework. The cell is called rnnCell_dualNet_, and it takes a number of parameters to define its behavior.
    
    Parameters
    ----------
    n_neuron: int
        The number of neurons in the RNN cell.
    
    n_peptide: int
        The number of peptides in the RNN cell.
    
    dt: float
        The time step of the RNN cell.
    
    weight_train: List[bool]
        A list of flags indicating which weights should be trained.
    
    initializer: str
        The initializer to use for the weights.
    
    log_train: bool
        A flag indicating whether to apply log transformation to the input for training.
    
    n_genes: int
        The number of genes in the RNN cell. genes are used for sample-specific modulation (e.g. pc2)
    
    ablate: bool
        A flag indicating whether to apply ablation to the neurons in the RNN cell.
    
    synapse_mask: bool
        A flag indicating whether to apply a mask to the synapse matrix.
    
    positive_input: bool
        A flag indicating whether to apply a positive constraint to the input layer.
    
    return_firing: bool
        A flag indicating whether to return the firing rates of the neurons.
    
    noisy_fire: bool
        A flag indicating whether to apply noise to the firing rates of the neurons.
    
    magnitude_fit: bool
        A flag indicating whether to fit the magnitudes of the weights while keeping their relative values constant.
    
    connectivity_mask: bool
        A flag indicating whether to apply a mask to the connectivity matrix.
    
    synapse_reg: str
        The regularizer to apply to the synapse matrix.
    
    noise: str
        The noise model to apply to the firing rates of the neurons.
    
    
    Attributes
    ----------
    n_neuron: int
        The number of neurons in the RNN cell.
    n_peptide: int
        The number of peptides in the RNN cell.
    n_genes: int
        The number of genes in the RNN cell.
    ablate: bool
        A flag indicating whether to apply ablation to the RNN cell.
    synapse_mask: bool
        A flag indicating whether to apply a mask to the synapse matrix.
    connectivity_mask: bool
        A flag indicating whether to apply a mask to the connectivity matrix.
    init: str
        The initializer to use for the weights.
    weight_train: List[bool]
        A list of flags indicating which weights should be trained.
    dt: float
        The time step of the RNN cell.
    log_train: bool
        A flag indicating whether to apply log transformation to the input for training.
    positive_input: bool
        A flag indicating whether to apply a positive constraint to the input layer.
    return_firing: bool
        A flag indicating whether to return the firing rates of the neurons.
    noisy_fire: bool
        A flag indicating whether to apply noise to the firing rates of the neurons.
    magnitude_fit: bool
        A flag indicating whether to fit the magnitudes of the weights while keeping their relative values constant.
    synapse_reg: Regularizer
        The regularizer to apply to the synapse matrix
    noise: str
        The noise model to apply to the firing rates of the neurons.
    
    D: tf.Variable
        The extracellular diffusion constant of each neuropeptide.
    pep_prod_rate: tf.Variable
        The production rate of the peptides.
    pep_decay_rate: tf.Variable
        The decay rate of the peptides.
    pep_action: tf.Variable
        The action of the peptides on the neurons.
    synapse_matrix: tf.Variable
        The matrix of synapses connecting the neurons.
    neuron_decay: tf.Variable
        The decay rate of the neurons.
    input_layer: tf.Variable
        The input layer of the RNN cell (stimulus->neurons).
    output_layer: tf.Variable
        Output
    """

    #121321--Full pattern competition state model
    def __init__(self, n_neuron=10,n_peptide=3,dt=1/120, weight_train=[True for i in range(11)],
                 initializer='ones',log_train=False,n_genes=0,ablate=False,synapse_mask=False,
                 positive_input=False,return_firing=True,noisy_fire=True,
                 magnitude_fit=False,connectivity_mask=False,
                 synapse_reg=None,noise='exp',**kwargs):

        self.n_neuron = n_neuron
        self.n_peptide = n_peptide
        self.n_genes = n_genes
        self.ablate = ablate

        self.units = self.n_neuron #+ self.n_neuron*self.n_peptide
        self.state_size = self.n_neuron + self.n_neuron*self.n_peptide +self.n_genes
        if self.ablate:
            self.state_size = self.state_size + self.n_neuron
        self.synapse_mask = synapse_mask #sample-wise mask (use if using synapse-based RNAi)
        if synapse_mask:
            self.state_size += self.n_neuron
        self.connectivity_mask = connectivity_mask #permanent mask to use if training syn weights
            #^Let's you define connectivity w/o defining weights

        self.init=initializer
        self.weight_train = weight_train
        self.dt = dt
        self.log_train=log_train
        self.positive_input = positive_input
        self.return_firing = return_firing
        self.noisy_fire=noisy_fire
        self.magnitude_fit = magnitude_fit #use this to train pep+synapse while keeping the relative vallues constant
        self.synapse_reg = keras.regularizers.get(synapse_reg) #regularizer to apply to synapse matrix
        self.noise = noise #firing rate noise model
        super(rnnCell_dualNet_, self).__init__(**kwargs)

    def build(self, input_shape,):
        self.D = self.add_weight(name='D',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[0])
        self.pep_prod_rate = self.add_weight(name='pep_product_rate',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[1])
        self.pep_decay_rate = self.add_weight(name='pep_decay_rate',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[2])
        self.pep_action = self.add_weight(name='pep_action',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[3])
        self.synapse_matrix = self.add_weight(name='synapse_matrix',
                                    shape=(self.n_neuron,self.n_neuron,),
                                    initializer=self.init,
                                    trainable=self.weight_train[4],
                                    regularizer = self.synapse_reg)  # <--apply the synapse regularizer
        self.neuron_decay = self.add_weight(name='neuron_decay',
                                    shape=(1,),
                                    initializer=self.init,
                                    trainable=self.weight_train[5])
        self.input_layer = self.add_weight(name='input_layer',
                                    shape=(self.n_neuron,),
                                    initializer=self.init,
                                    trainable=self.weight_train[6])
        if self.magnitude_fit:
            self.s_scale = self.add_weight(name='synapse_scale',
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)
            self.p_scale = self.add_weight(name='peptide_scale',
                                    shape=(1,),
                                    initializer='ones',
                                    trainable=True)
        if self.connectivity_mask:
            self.connectivity = self.add_weight(name='connectivity',
                                    shape=(self.n_neuron,self.n_neuron,),
                                    initializer='ones',
                                    trainable=False)

        super(rnnCell_dualNet_, self).build(input_shape)

    def get_config(self):
        config = {'dt': self.dt,
                  'n_neuron': self.n_neuron,
                  'n_peptide': self.n_peptide,
                  'weight_train': self.weight_train,
                  'log_train': self.log_train,
                  'n_genes': self.n_genes,
                  'ablate':self.ablate,
                  'synapse_mask':self.synapse_mask,
                  'positive_input':self.positive_input,
                  'return_firing':self.return_firing,
                  'noisy_fire':self.noisy_fire,
                  'magnitude_fit':self.magnitude_fit,
                  'connectivity_mask':self.connectivity_mask,
                  'synapse_reg':self.synapse_reg,
                  'noise':self.noise,
                }
        base_config = super(rnnCell_dualNet_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self,inputs, states,):
        #name parameters
        if self.log_train:
            func = tf.exp
        else:
            func = tf.abs
        pep_prod_rate=func(self.pep_prod_rate)
        pep_decay_rate=func(self.pep_decay_rate)
        neuron_decay=func(self.neuron_decay)
        D=func(self.D)
        dt = self.dt

        #pull out states
        u = inputs#[:,0]
        ablate_shift = 0 #keeps track where youre pulling out
        if self.ablate:
            ablate_mat = states[0][:,:self.n_neuron]
            ablate_shift = self.n_neuron
        if self.synapse_mask:
            syn_mask = states[0][:,ablate_shift:ablate_shift+self.n_neuron]
            ablate_shift += self.n_neuron
        if self.n_genes>0:
            pc2=states[0][:,0+ablate_shift][:,None]
        neuron = states[0][:,self.n_genes+ablate_shift:self.n_neuron+self.n_genes+ablate_shift]
        peptide = K.reshape(states[0][:,self.n_genes+self.n_neuron+ablate_shift:],(-1,self.n_neuron,self.n_peptide))
        firing = K.relu(neuron)
        # print('SHAPES',ablate_mat.shape,pc2.shape,neuron.shape,peptide.shape,states[0].shape)


        #run operations
        if self.ablate:
            firing = firing * ablate_mat
            neuron = neuron * ablate_mat
        # print(neuron.shape,firing.shape,peptide.shape)
#        firing = tf.random.stateless_poisson(firing.shape,see=[0,1],lam=firing)
        if self.noisy_fire:
            if self.noise =='exp':
                firing=firing+1e-2
                firing = -firing*tf.log(1-K.random_uniform(K.shape(firing)))  #exponential sampling
            else:
                firing = K.clip(firing,0,10) + K.random_normal(K.shape(firing)) * self.noise
        firing = K.clip(firing,0,10) # 020422
##        print('FIRING',firing.shape)
#        f_sum = K.sum(firing,axis=1,keepdims=True)
#        firing=firing/f_sum
        #peptide production and decay
        pep_prod = pep_prod_rate * firing[:,:,None]
        pep_decay = pep_decay_rate * peptide
        #peptide diffusion
        sq = int(self.n_neuron**.5)
        # print(peptide.shape)
        peptide_space = K.reshape(peptide,(-1,sq,sq,self.n_peptide))
        print('pep_space',peptide_space.shape)
        pep_diffuse = D*(tf.roll(peptide_space,-1,axis=1) +
                                       tf.roll(peptide_space,1,axis=1) +
                                       tf.roll(peptide_space,-1,axis=2) +
                                       tf.roll(peptide_space,1,axis=2) -
                                       4*peptide_space)
        pep_diffuse = K.reshape(pep_diffuse,(-1,self.n_neuron,self.n_peptide))

#       #neuron changes
        #manage magnitude fitting parameters
        if self.magnitude_fit:
            pep_action = self.pep_action*self.p_scale
            s_scale = self.s_scale
        else:
            pep_action = self.pep_action
            s_scale = 1
        #apply neuron changes
        d_neuron = K.sum(peptide*pep_action,axis=-1) * pc2
        if self.synapse_mask: #synapse mask selectively removes synaptic output of neuron in sample
            if self.positive_input:
                syn_neuron = (firing*syn_mask) @ (K.abs(self.synapse_matrix))
            else:
                syn_neuron = (firing*syn_mask) @ (self.synapse_matrix)
        else:
            if self.positive_input:
                syn_neuron = firing @ (K.abs(self.synapse_matrix))
            else:
                syn_neuron = firing @ (self.synapse_matrix)
        d_neuron += syn_neuron * s_scale #s_scale must be applied after firing@synapse to avoid huge memory issue (probably something to do with gradient saving?)
        d_neuron -= neuron * neuron_decay
#        print(u.shape,d_neuron.shape,self.input_layer.shape,(self.input_layer*u).shape)
        d_neuron += self.input_layer*u
        d_peptide = pep_prod - pep_decay + pep_diffuse
        d_peptide = K.reshape(d_peptide,(-1,self.n_neuron*self.n_peptide))

        pep_new = states[0][:,self.n_genes+self.n_neuron+ablate_shift:] + d_peptide*dt
        neuron_new = neuron + d_neuron*dt
        #pack and ship
        X_new = []
        if self.ablate:
            neuron_new = neuron_new*ablate_mat
            X_new.append(ablate_mat)
        if self.synapse_mask:
            X_new.append(syn_mask)
        if self.n_genes>0:
            X_new.append(pc2)
        X_new.extend([neuron_new,pep_new])
        X_new = K.concatenate(X_new,axis=-1)
        if self.return_firing=='both':
            return_vec = K.concatenate([firing,neuron_new])
            return return_vec, [X_new,]
        if self.return_firing:
            return firing,[X_new,]
        else:
            return neuron_new,[X_new,]














'''
CORRELATION APPROXIMATOR
'''
class cov_aprox_method1(keras.layers.Layer):
    #leaky integrator with normalization of firing rate at each timestep
    def __init__(self, units,beta, **kwargs):
        self.units = units
        self.state_size = units
        self.beta = beta
        super(cov_aprox_method1, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        old_C = states[0] * (1-self.beta)
        f_hat = self.beta * inputs/K.sum(inputs,axis=1,keepdims=True)
        print('FHAT:',f_hat.shape,K.sum(inputs,axis=1,keepdims=True))
        new_C = old_C + f_hat
        return new_C, [new_C]

class cov_aprox_method2(keras.layers.Layer):
    #leaky integrator with normalization of firing rate at each timestep
    def __init__(self, units,beta, **kwargs):
        self.units = units
        self.state_size = units
        self.beta = beta
        super(cov_aprox_method2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        old_C = states[0] * (1-self.beta)
        f_hat = self.beta * inputs
        print('FHAT:',f_hat.shape,inputs.shape)
        new_C = old_C + f_hat
        return new_C/K.sum(inputs,axis=1,keepdims=True), [new_C]

class cov_aprox_method3(keras.layers.Layer):
    #leaky integrator with normalization of firing rate at each timestep
    def __init__(self, units,beta, **kwargs):
        self.units = units
        self.state_size = units
        self.beta = beta
        super(cov_aprox_method3, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        old_C = states[0] * (1-self.beta)
        f_hat = self.beta * (inputs-K.mean(inputs,axis=1,keepdims=True))
        print('FHAT:',f_hat.shape,inputs.shape)
        new_C = old_C + f_hat
        return new_C, [new_C]





"""
class rnnCell_dualNet_(keras.layers.Layer):
    #121321--Full pattern competition state model
    def __init__(self, n_neuron=10,n_peptide=3,dt=1/120, weight_train=[True for i in range(11)],
                 initializer='ones',log_train=False,n_genes=0, **kwargs):


        self.n_peptide = n_peptide
        self.n_genes = n_genes
        self.n_neuron = n_neuron
        self.units = self.n_neuron #+ self.n_neuron*self.n_peptide
        self.state_size = self.n_neuron + self.n_neuron*self.n_peptide +self.n_genes
        self.init=initializer
        self.weight_train = weight_train
        self.dt = dt
        self.log_train=log_train
        super(rnnCell_dualNet_, self).__init__(**kwargs)

    def build(self, input_shape,):
        self.D = self.add_weight(name='D',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[0])
        self.pep_prod_rate = self.add_weight(name='pep_product_rate',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[1])
        self.pep_decay_rate = self.add_weight(name='pep_decay_rate',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[2])
        self.pep_action = self.add_weight(name='pep_action',
                                    shape=(self.n_peptide,),
                                    initializer=self.init,
                                    trainable=self.weight_train[3])
        self.synapse_matrix = self.add_weight(name='synapse_matrix',
                                    shape=(self.n_neuron,self.n_neuron,),
                                    initializer=self.init,
                                    trainable=self.weight_train[4])
        self.neuron_decay = self.add_weight(name='neuron_decay',
                                    shape=(1,),
                                    initializer=self.init,
                                    trainable=self.weight_train[5])
        self.input_layer = self.add_weight(name='input_layer',
                                    shape=(self.n_neuron,),
                                    initializer=self.init,
                                    trainable=self.weight_train[6])

        super(rnnCell_dualNet_, self).build(input_shape)

    def get_config(self):
        config = {'dt': self.dt,
                  'weight_train': self.weight_train,
                  'log_train': self.log_train,
                  'n_genes': self.n_genes
                }
        base_config = super(rnnCell_dualNet_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self,inputs, states):
        #name parameters
        if self.log_train:
            func = tf.exp
        else:
            func = tf.abs
        pep_prod_rate=func(self.pep_prod_rate)
        pep_decay_rate=func(self.pep_decay_rate)
        neuron_decay=func(self.neuron_decay)
        D=func(self.D)
        dt = self.dt

        #pull out states
        u = inputs#[:,0]
        if self.n_genes>0:
            pc2=states[0][:,0][:,None]
        neuron = states[0][:,self.n_genes:self.n_neuron+self.n_genes]
        peptide = K.reshape(states[0][:,self.n_genes+self.n_neuron:],(-1,self.n_neuron,self.n_peptide))
        firing = K.relu(neuron)+1e-3
        print(neuron.shape,firing.shape,peptide.shape)
#        firing = tf.random.stateless_poisson(firing.shape,see=[0,1],lam=firing)
        firing = -firing*tf.log(1-K.random_uniform(K.shape(firing)))
        #peptide production and decay
#        print(firing[:,:,None].shape,pep_prod_rate.shape,pep_decay_rate.shape)
        pep_prod = pep_prod_rate * firing[:,:,None]
        pep_decay = pep_decay_rate * peptide
        #peptide diffusion
        sq = int(self.n_neuron**.5)
        print(peptide.shape)
        peptide_space = K.reshape(peptide,(-1,sq,sq,self.n_peptide))
        print('pep_space',peptide_space.shape)
        pep_diffuse = D*(tf.roll(peptide_space,-1,axis=1) +
                                       tf.roll(peptide_space,1,axis=1) +
                                       tf.roll(peptide_space,-1,axis=2) +
                                       tf.roll(peptide_space,1,axis=2) -
                                       4*peptide_space)
        pep_diffuse = K.reshape(pep_diffuse,(-1,self.n_neuron,self.n_peptide))
#
#        #neuron changes
        d_neuron = K.sum(peptide*self.pep_action,axis=-1) * pc2
#        print((firing @self.synapse_matrix).shape)
        d_neuron +=  firing @self.synapse_matrix
        d_neuron -= neuron * neuron_decay
        print(u.shape,d_neuron.shape,self.input_layer.shape,(self.input_layer*u).shape)
        d_neuron += self.input_layer*u
#
        d_peptide = pep_prod - pep_decay + pep_diffuse
        d_peptide = K.reshape(d_peptide,(-1,self.n_neuron*self.n_peptide))

        pep_new = states[0][:,self.n_neuron+self.n_genes:] + d_peptide*dt
        neuron_new = neuron + d_neuron*dt
        if self.n_genes==0:
            X_new = K.concatenate([neuron_new,pep_new],axis=-1)
        else:
            X_new = K.concatenate([pc2,neuron_new,pep_new],axis=-1)
        return neuron_new,[X_new,]


      """
