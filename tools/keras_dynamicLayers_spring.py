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

class rnnCell_spring_(keras.layers.Layer):
    #rnn cell that models the dynamics of a linear spring. Overkill for fitting but a simple demo
    #seeing if can get a reasonable fit for the interaction models --113021
    def __init__(self, units=2, dt=1/120, weight_train=[True for i in range(9)], h=1,
                 initializer=keras.initializers.glorot_uniform(),
                 log_train=False,**kwargs):
        self.dt = dt
        self.weight_train = weight_train
        self.init = initializer
        self.units = units
        self.state_size = units
        self.log_train = log_train
        super(rnnCell_spring_, self).__init__(**kwargs)
    
    def build(self,input_shape,):
        self._k = self.add_weight(name='k', 
                                    shape=(1,),
                                    initializer=self.init,
                                    trainable=self.weight_train[0])
        self._b = self.add_weight(name='b', 
                                    shape=(1,),
                                    initializer=self.init,
                                    trainable=self.weight_train[1])
        self._input_mat = self.add_weight(name='input_mat', 
                                    shape=(2,2,),
                                    initializer=self.init,
                                    trainable=self.weight_train[2])
        self._input_mask = self.add_weight(name='input_mask', 
                                    shape=(2,2,),
                                    initializer='ones',
                                    trainable=False)
        super(rnnCell_spring_, self).build(input_shape)
    
    def get_config(self):
        config = {'dt': self.dt,
                  'weight_train': self.weight_train,
                  'log_train': self.log_train,
                }
        base_config = super(rnnCell_spring_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self,inputs, states):
        #name parameters
        if self.log_train:
            func = tf.exp
        else:
            func = tf.abs
        k=func(self._k)
        b=func(self._b)
        input_mat = self._input_mat *self._input_mask
        dt = self.dt   
        
        #pull out states
        u = inputs#[:,0]
        X = states[0]
        print('in,state',inputs.shape,states[0].shape)
        x = X[:,0]
        v = X[:,1]
#         print('u,z_t',u.shape,z.shape)
        #dynamic update of state
        x_new = x+(v*dt)
        v_new = v+(-k*x-b*v)*dt
        
        #package back together
        X_new = tf.stack([x_new,v_new],axis=-1)
        print('pre-op',X_new.shape)
        print('input-op',tf.matmul(u,input_mat).shape)
        X_new = X_new + tf.matmul(u,input_mat)*dt
        print('X_new',X_new.shape)
        return X_new, [X_new,]