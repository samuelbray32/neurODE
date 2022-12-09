#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:52:05 2022

@author: sam
"""

import keras
from keras import backend as K
from .keras_dynamicLayers import initial_condition_layer, trainableInput
from .keras_dynamicLayers_RNNcell_dualModel import rnnCell_dualNet_ as rnnCell_
import tensorflow as tf
import numpy as np

def get_n_neuron(model,latent_model=False):
    if latent_model: #use this to extract value from a latent model
        return model.layers[-2].weights[-1].shape.as_list()[0]
    return model.layers[-4].weights[-1].shape.as_list()[0]

def ablate_square(ablation,n_neuron,n_test):
    sq = int(n_neuron**.5)
    u0 = np.ones((sq,sq))
    w = int((n_neuron*ablation)**.5)
    if w==0:
        h=0
    else:
        h= int((n_neuron*ablation)/w)
    u0[:w,:h] = 0
    u_ablate = []
    for _ in range(n_test):
        roll = np.random.randint(0,sq,2)
        u_ii = u0.copy()
        u_ii = np.roll(np.roll(u_ii,roll[0],axis=0),roll[1],axis=1)
        u_ablate.append(u_ii)
    u_ablate = np.reshape(np.array(u_ablate),(n_test,n_neuron))
    return u_ablate


def synapse_mask_fun(ID,n_neuron=2500,fractions=np.array([0,0,.125,.125,.5,.125]),
                    gene_order = ['WT','pc2','tbh','gad','chat','th',]):
                    #note: gene order default and fractions are those used in robustness paper
        if type(ID) is str:
            for i,gene in enumerate(gene_order):
                if gene==ID:
                    ID=i
        if not type(ID) is list:
            ID = [ID]
        mask = []
        ind = np.arange(n_neuron)
        np.random.seed(0)
        np.random.shuffle(ind)
        loc=0
        for f in fractions:
            n = int(f*n_neuron)
            x = np.ones(n_neuron)
            x[ind[loc:loc+n]] = 0
            mask.append(x.copy())
            loc += n
        return (np.array(mask)[ID]).min(axis=0)

class zeroConstraint(keras.constraints.Constraint):
  """Constrains weight tensors to sum to 0."""
  def __init__(self,):
      return
  def __call__(self, w):
    mean = tf.reduce_mean(w)
    return w - mean

def map_weights(old, new):
    #used when changing and recompiling with new trainability
    #maps weight indexz of old model to index in new model
    names=['pep_decay_rate','synapse_matrix','neuron_decay','input_layer',
           'synapse_scale','peptide_scale','D','pep_product_rate','pep_action',
           'connectivity',]
    weight_map = []
    for nm in names:
        i_new = None
        for i,w in enumerate(new):
            if nm in w.name:
                i_new=i
                break
        if i_new is None: continue
        i_old = None
        for i,w in enumerate(old):
            if nm in w.name:
                i_old=i
                break
        if i_old is None:
            print (f'ERROR: {nm} not found in original weights')
            continue
        weight_map.append((i_old, i_new))
    return weight_map



def augment_model_trainability(model):
    #make new fully trainable RNN cell
    THETA_old = model.layers[-4]
    train_new = [True for _ in THETA_old.cell.weight_train]
    THETA_new_cell = rnnCell_(weight_train=train_new, initializer=THETA_old.cell.init,
                              log_train=THETA_old.cell.log_train, n_neuron=THETA_old.cell.n_neuron,
                             n_peptide=THETA_old.cell.n_peptide, n_genes=THETA_old.cell.n_genes,
                             ablate=THETA_old.cell.ablate, return_firing=THETA_old.cell.return_firing,
                             magnitude_fit=THETA_old.cell.magnitude_fit,
                             connectivity_mask=THETA_old.cell.connectivity_mask,
                             synapse_mask=THETA_old.cell.synapse_mask, noise=THETA_old.cell.noise)
    print(THETA_new_cell.weight_train, train_new)
    THETA_new = keras.layers.RNN(THETA_new_cell,return_sequences=True,return_state=True)

    #make new model
    U_in, U_in_gene, U_ablate, U_syn = model.inputs
    Q0 = model.layers[7].output
    U_eq = model.layers[6].output
    Q0 = THETA_new(U_eq,initial_state=[Q0,],)[1]
    Q_hat,Q_final = THETA_new(U_in,initial_state=[Q0,])
    Z = model.layers[-3](Q_hat)
    Z = model.layers[-2](Z)
    Z = model.layers[-1](Z)
    def my_loss(q0,qf):
        def my_loss_(y_true,y_pred):
            eq_loss = 0*keras.losses.mean_squared_error(q0,qf)
            print('eq_loss',eq_loss.shape)
            return keras.losses.mean_squared_error(y_true,y_pred) + eq_loss
    #        return keras.losses.mean_absolute_error(y_true,y_pred) + eq_loss
            #    return keras.losses.mean_absolute_percentage_error(y_true,y_pred)
        return my_loss_
    model_new = keras.models.Model(inputs=model.inputs, outputs=[Z])
    opt = keras.optimizers.Adam(learning_rate=1e-3)#keras.optimizers.Adadelta(learning_rate=1e-3)#
    model_new.compile(optimizer=opt,loss='mse')

    #set weights of new RNN
    w_old = THETA_old.get_weights()
    w_map = map_weights(THETA_old.weights,THETA_new.weights)
    w_new = [None for _ in w_map]
    for m in w_map:
        w_new[m[1]] = w_old[m[0]]
    model_new.layers[-4].set_weights(w_new)
    return model_new




def load_model(folder, return_latent_model=False):
    # loads model from file
    try:
        model = keras.models.load_model(f'{folder}trained_model',
            custom_objects={'trainableInput':trainableInput,'rnnCell_dualNet_':rnnCell_})
    except:
        model = keras.models.load_model(f'{folder}model',
                custom_objects={'trainableInput':trainableInput,'rnnCell_dualNet_':rnnCell_})


    if return_latent_model:
        #use the pull_latent_model func. rather than load from drive so weights are shared
        latent_model =  pull_latent_model(model)
        return model, latent_model
    return model

def save_model(folder,model,latent_model=None,note=None):
    #standardized model saving
    import os
    if not os.path.isdir(folder):
        os.makedirs(folder)
    model.save(folder+'model')
    if latent_model is None:
        latent_model = pull_latent_model(model)
    if not note is None:
        with open(f'{folder}notes.txt', 'w+') as fh:
            fh.write(note)
    latent_model.save(folder+'latent_model')
    return


def pull_latent_model(model):
    #pulls a latent model simulator from the model produced by the buildModel function in this file
    return keras.models.Model(model.inputs,model.layers[-2].input)



def build_model(log_train = False,n_neuron=50**2,
                n_peptide=2,init='glorot_uniform',trainable = [False for i in range(7)],
                magnitude_fit=True,n_genes=1,ablate=True,constrain_measure=False,
                sigmoid_measure=False,connectivity_mask = True,synapse_mask=True,
                noise = 1e-2):

    sq=int(n_neuron**.5)
    #stimulus
    U = keras.layers.Input((None,1,),name='U_in')
    #Model RNN
    THETA_cell = rnnCell_(initializer=init,log_train=log_train, weight_train=trainable,
                          n_neuron=n_neuron,n_peptide=n_peptide,n_genes=n_genes,ablate=ablate,
                          return_firing=True,magnitude_fit=magnitude_fit,
                          connectivity_mask=connectivity_mask, synapse_mask=synapse_mask,noise=noise)
    THETA = keras.layers.RNN(THETA_cell,return_sequences=True,return_state=True)

    #Initial condition generator
    U_dummy = keras.layers.Lambda(lambda x: x[:,0,:])(U)
    if log_train:
        initial_condition = trainableInput(p_shape=(THETA_cell.state_size-n_genes-2*n_neuron,),initializer='zeros')
        Q0 = initial_condition(U_dummy)
        Q0 = keras.layers.Lambda(lambda x: tf.exp(x))(Q0)
    else:
        initial_condition = trainableInput(p_shape=(THETA_cell.state_size-n_genes-2*n_neuron,),initializer=init)
        Q0 = initial_condition(U_dummy)
    #    Q0 = keras.layers.Lambda(lambda x: tf.abs(x))(Q0)

    #genetic input
    U_gene = keras.layers.Input((n_genes,),name='U_in_gene')
    U_ablate = keras.layers.Input((n_neuron,),name='U_ablate')
    U_syn = keras.layers.Input((n_neuron,),name='U_syn')
    Q0 = keras.layers.Concatenate(axis=1)([U_ablate,U_syn,U_gene,Q0])

    '''
    # if trainable individ peptides
    if extra_pep:
        U_pep = keras.layers.Input((n_peptide,),name='U_in_gene')
        Q0 = keras.layers.Concatenate(axis=1)([U_ablate,U_syn,U_gene,U_pep,Q0])
    '''

    #run equilibrium and real model
    U_eq = keras.layers.Lambda(lambda x: 0*x[:,:1200])(U)
    Q0 = THETA(U_eq,initial_state=[Q0,],)[1]
    Q_hat,Q_final = THETA(U,initial_state=[Q0,])

    #M = keras.layers.Lambda(lambda x: tf.expand_dims(K.transpose(tf.gather(K.transpose(x),0)),axis=-1))
    M = keras.layers.Lambda(lambda x: K.relu(x[:,:,:n_neuron]))
    #M_reg_cell = cov_aprox(units=n_neuron,beta=.05)
    #M_reg = keras.layers.RNN(M_reg_cell,return_sequences=True,)
    #M_reg = keras.layers.Lambda(lambda x: x/K.sqrt(K.sum(K.square(x),axis=2,keepdims=True)))
    #M_reg = keras.layers.Lambda(lambda x: x/K.sum(x,axis=2,keepdims=True))

    act = 'linear'
    if sigmoid_measure:
        act = 'sigmoid'
    constrain = None
    if constrain_measure:
        constrain = zeroConstraint()
    M2 = keras.layers.Conv1D(filters=1,kernel_size=1,input_shape=(None,n_neuron,),activation=act,
                             kernel_initializer='glorot_uniform',data_format='channels_last',
                             kernel_constraint=constrain,kernel_regularizer='l2')
    Z_hat = M(Q_hat)
    Z_hat = M2(Z_hat)
    Z_hat = keras.layers.Lambda(lambda x: K.squeeze(x,-1))(Z_hat)

    #In[]
    #full model
    def my_loss(q0,qf):
        def my_loss_(y_true,y_pred):
            eq_loss = 0*keras.losses.mean_squared_error(q0,qf)
            print('eq_loss',eq_loss.shape)
            return keras.losses.mean_squared_error(y_true,y_pred) + eq_loss
    #        return keras.losses.mean_absolute_error(y_true,y_pred) + eq_loss
            #    return keras.losses.mean_absolute_percentage_error(y_true,y_pred)
        return my_loss_

    model = keras.models.Model(inputs=[U,U_gene,U_ablate,U_syn],outputs=[Z_hat])
    opt = keras.optimizers.Adam(learning_rate=1e-3)#keras.optimizers.Adadelta(learning_rate=1e-3)#
    model.compile(optimizer=opt,loss='mse')#my_loss(Q0,Q_final),)
    #model that leaves it in latent space
    latent_model = keras.models.Model(inputs=[U,U_gene,U_ablate,U_syn], outputs=[Q_hat])
    return model, latent_model, THETA
