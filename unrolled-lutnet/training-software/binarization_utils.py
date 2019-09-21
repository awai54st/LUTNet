import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.layers.normalization import BatchNormalization
from tensorflow.python.framework import ops
#from multi_gpu import make_parallel

def binarize(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,-1,1)
    rounded = K.sign(clipped)
    return clipped + K.stop_gradient(rounded - clipped)

class Residual_sign(Layer):
    def __init__(self, levels=1,trainable=True,**kwargs):
        self.levels=levels
        self.trainable=trainable
        super(Residual_sign, self).__init__(**kwargs)
    def build(self, input_shape):
        ars=np.arange(self.levels)+1.0
        ars=ars[::-1]
        means=ars/np.sum(ars)
        #self.means=[K.variable(m) for m in means]
        #self.trainable_weights=self.means
        self.means = self.add_weight(name='means',
            shape=(self.levels, ),
            initializer=keras.initializers.Constant(value=means),
            trainable=self.trainable) # Trainable scaling factors for residual binarisation
    def call(self, x, mask=None):
        resid = x
        out_bin=0

        if self.levels==1:
            for l in range(self.levels):
                #out=binarize(resid)*K.abs(self.means[l])
                out=binarize(resid)*abs(self.means[l])
                #out_bin=out_bin+out
                out_bin=out_bin+out#no gamma per level
                resid=resid-out
        elif self.levels==2:
            out=binarize(resid)*abs(self.means[0])
            out_bin=out
            resid=resid-out
            out=binarize(resid)*abs(self.means[1])
            out_bin=tf.stack([out_bin,out])
            resid=resid-out
                
        return out_bin

    def get_output_shape_for(self,input_shape):
        if self.levels==1:
            return input_shape
        else:
            return (self.levels, input_shape)
    def compute_output_shape(self,input_shape):
        if self.levels==1:
            return input_shape
        else:
            return (self.levels, input_shape)
    def set_means(self,X):
        means=np.zeros((self.levels))
        means[0]=1
        resid=np.clip(X,-1,1)
        approx=0
        for l in range(self.levels):
            m=np.mean(np.absolute(resid))
            out=np.sign(resid)*m
            approx=approx+out
            resid=resid-out
            means[l]=m
            err=np.mean((approx-np.clip(X,-1,1))**2)

        means=means/np.sum(means)
        sess=K.get_session()
        sess.run(self.means.assign(means))

class binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,strides=(1,1),levels=1,pruning_prob=0,first_layer=False,LUT=True,BINARY=True,**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		if padding=='valid':
			self.PADDING = "VALID"
		elif padding=='same':
			self.PADDING = "SAME"
		self.strides=strides
		self.levels=levels
		self.first_layer=first_layer
		self.LUT=LUT
		self.BINARY=BINARY
		self.window_size=self.ch_in*self.k*self.k
		#self.rand_map=np.random.randint(self.window_size, size=[self.window_size, 1]) # Randomisation map for subsequent input connections
		super(binary_conv,self).__init__(**kwargs)
	def build(self, input_shape):
		self.rand_map_0 = self.add_weight(name='rand_map_0', 
			shape=(self.window_size, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
			trainable=False) # Randomisation map for subsequent input connections
		self.rand_map_1 = self.add_weight(name='rand_map_1', 
			shape=(self.window_size, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
			trainable=False) # Randomisation map for subsequent input connections
		self.rand_map_2 = self.add_weight(name='rand_map_2', 
			shape=(self.window_size, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
			trainable=False) # Randomisation map for subsequent input connections

		stdv=1/np.sqrt(self.k*self.k*self.ch_in)
		self.gamma=K.variable(1.0)
#		if self.first_layer==True:
#			w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#			w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#			w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#			w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#
#			self.w1=K.variable(w1)
#			self.w2=K.variable(w2)
#			self.w3=K.variable(w3)
#			self.w4=K.variable(w4)
#			self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.gamma]

		if self.levels==1 or self.first_layer==True:
			w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
			self.w=K.variable(w)
			self.trainable_weights=[self.w,self.gamma]
		elif self.levels==2:
			if self.LUT==True:

				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w9 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w10 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w11 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w12 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w13 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w14 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w15 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w16 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w17 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w18 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w19 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w20 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w21 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w22 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w23 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w24 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w25 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w26 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w27 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w28 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w29 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w30 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w31 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w32 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)

				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)
				self.w9=K.variable(w9)
				self.w10=K.variable(w10)
				self.w11=K.variable(w11)
				self.w12=K.variable(w12)
				self.w13=K.variable(w13)
				self.w14=K.variable(w14)
				self.w15=K.variable(w15)
				self.w16=K.variable(w16)
				self.w17=K.variable(w17)
				self.w18=K.variable(w18)
				self.w19=K.variable(w19)
				self.w20=K.variable(w20)
				self.w21=K.variable(w21)
				self.w22=K.variable(w22)
				self.w23=K.variable(w23)
				self.w24=K.variable(w24)
				self.w25=K.variable(w25)
				self.w26=K.variable(w26)
				self.w27=K.variable(w27)
				self.w28=K.variable(w28)
				self.w29=K.variable(w29)
				self.w30=K.variable(w30)
				self.w31=K.variable(w31)
				self.w32=K.variable(w32)

				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.w9,self.w10,self.w11,self.w12,self.w13,self.w14,self.w15,self.w16,self.w17,self.w18,self.w19,self.w20,self.w21,self.w22,self.w23,self.w24,self.w25,self.w26,self.w27,self.w28,self.w29,self.w30,self.w31,self.w32,self.gamma]

			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]
	

		elif self.levels==3:
			if self.LUT==True:
				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)
				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.gamma]
			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]

		self.pruning_mask = self.add_weight(name='pruning_mask',
			shape=(self.window_size,self.nfilters),
			initializer=keras.initializers.Constant(value=np.ones((self.window_size,self.nfilters))),
			trainable=False) # LUT pruning based on whether inputs get repeated



#		if keras.backend._backend=="mxnet":
#			w=w.transpose(3,2,0,1)

#		if self.levels==1:#train baseline with no resid gamma scaling
#			w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#			self.w=K.variable(w)
#			self.trainable_weights=[self.w,self.gamma]
#		elif self.levels==2:
#			w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
#			self.w=K.variable(w)
#			self.trainable_weights=[self.w,self.gamma]



	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)

		if self.levels==1 or self.first_layer==True:
			if self.BINARY==False:
				self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
			else:
				self.clamped_w=constraint_gamma*binarize(self.w)
		elif self.levels==2:
			if self.LUT==True:
				if self.BINARY==False:
					self.clamped_w1=constraint_gamma*K.clip(self.w1,-1,1)
					self.clamped_w2=constraint_gamma*K.clip(self.w2,-1,1)
					self.clamped_w3=constraint_gamma*K.clip(self.w3,-1,1)
					self.clamped_w4=constraint_gamma*K.clip(self.w4,-1,1)
					self.clamped_w5=constraint_gamma*K.clip(self.w5,-1,1)
					self.clamped_w6=constraint_gamma*K.clip(self.w6,-1,1)
					self.clamped_w7=constraint_gamma*K.clip(self.w7,-1,1)
					self.clamped_w8=constraint_gamma*K.clip(self.w8,-1,1)
					self.clamped_w9=constraint_gamma*K.clip(self.w9,-1,1)
					self.clamped_w10=constraint_gamma*K.clip(self.w10,-1,1)
					self.clamped_w11=constraint_gamma*K.clip(self.w11,-1,1)
					self.clamped_w12=constraint_gamma*K.clip(self.w12,-1,1)
					self.clamped_w13=constraint_gamma*K.clip(self.w13,-1,1)
					self.clamped_w14=constraint_gamma*K.clip(self.w14,-1,1)
					self.clamped_w15=constraint_gamma*K.clip(self.w15,-1,1)
					self.clamped_w16=constraint_gamma*K.clip(self.w16,-1,1)
					self.clamped_w17=constraint_gamma*K.clip(self.w17,-1,1)
					self.clamped_w18=constraint_gamma*K.clip(self.w18,-1,1)
					self.clamped_w19=constraint_gamma*K.clip(self.w19,-1,1)
					self.clamped_w20=constraint_gamma*K.clip(self.w20,-1,1)
					self.clamped_w21=constraint_gamma*K.clip(self.w21,-1,1)
					self.clamped_w22=constraint_gamma*K.clip(self.w22,-1,1)
					self.clamped_w23=constraint_gamma*K.clip(self.w23,-1,1)
					self.clamped_w24=constraint_gamma*K.clip(self.w24,-1,1)
					self.clamped_w25=constraint_gamma*K.clip(self.w25,-1,1)
					self.clamped_w26=constraint_gamma*K.clip(self.w26,-1,1)
					self.clamped_w27=constraint_gamma*K.clip(self.w27,-1,1)
					self.clamped_w28=constraint_gamma*K.clip(self.w28,-1,1)
					self.clamped_w29=constraint_gamma*K.clip(self.w29,-1,1)
					self.clamped_w30=constraint_gamma*K.clip(self.w30,-1,1)
					self.clamped_w31=constraint_gamma*K.clip(self.w31,-1,1)
					self.clamped_w32=constraint_gamma*K.clip(self.w32,-1,1)

				else:

					self.clamped_w1=constraint_gamma*binarize(self.w1)
					self.clamped_w2=constraint_gamma*binarize(self.w2)
					self.clamped_w3=constraint_gamma*binarize(self.w3)
					self.clamped_w4=constraint_gamma*binarize(self.w4)
					self.clamped_w5=constraint_gamma*binarize(self.w5)
					self.clamped_w6=constraint_gamma*binarize(self.w6)
					self.clamped_w7=constraint_gamma*binarize(self.w7)
					self.clamped_w8=constraint_gamma*binarize(self.w8)
					self.clamped_w9=constraint_gamma*binarize(self.w9)
					self.clamped_w10=constraint_gamma*binarize(self.w10)
					self.clamped_w11=constraint_gamma*binarize(self.w11)
					self.clamped_w12=constraint_gamma*binarize(self.w12)
					self.clamped_w13=constraint_gamma*binarize(self.w13)
					self.clamped_w14=constraint_gamma*binarize(self.w14)
					self.clamped_w15=constraint_gamma*binarize(self.w15)
					self.clamped_w16=constraint_gamma*binarize(self.w16)
					self.clamped_w17=constraint_gamma*binarize(self.w17)
					self.clamped_w18=constraint_gamma*binarize(self.w18)
					self.clamped_w19=constraint_gamma*binarize(self.w19)
					self.clamped_w20=constraint_gamma*binarize(self.w20)
					self.clamped_w21=constraint_gamma*binarize(self.w21)
					self.clamped_w22=constraint_gamma*binarize(self.w22)
					self.clamped_w23=constraint_gamma*binarize(self.w23)
					self.clamped_w24=constraint_gamma*binarize(self.w24)
					self.clamped_w25=constraint_gamma*binarize(self.w25)
					self.clamped_w26=constraint_gamma*binarize(self.w26)
					self.clamped_w27=constraint_gamma*binarize(self.w27)
					self.clamped_w28=constraint_gamma*binarize(self.w28)
					self.clamped_w29=constraint_gamma*binarize(self.w29)
					self.clamped_w30=constraint_gamma*binarize(self.w30)
					self.clamped_w31=constraint_gamma*binarize(self.w31)
					self.clamped_w32=constraint_gamma*binarize(self.w32)

			else:
				if self.BINARY==False:
					self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
				else:
					self.clamped_w=constraint_gamma*binarize(self.w)

		if keras.__version__[0]=='2':

			if self.levels==1 or self.first_layer==True:
				self.out=K.conv2d(x, kernel=self.clamped_w*tf.reshape(self.pruning_mask, [self.k, self.k, self.ch_in, self.nfilters]), padding=self.padding,strides=self.strides )
			elif self.levels==2:
				if self.LUT==True:
					x0_patches = tf.extract_image_patches(x[0,:,:,:,:],
						[1, self.k, self.k, 1],
						[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
						padding=self.PADDING)
					x1_patches = tf.extract_image_patches(x[1,:,:,:,:],
						[1, self.k, self.k, 1],
						[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
						padding=self.PADDING)
					# Special hack for randomising the subsequent input connections: tensorflow does not support advanced matrix indexing
					x0_shuf_patches=tf.transpose(x0_patches, perm=[3, 0, 1, 2])
					x0_shuf_patches_0 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_0, tf.int32))
					x0_shuf_patches_1 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_1, tf.int32))
					x0_shuf_patches_2 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_2, tf.int32))
					x0_shuf_patches_0=tf.transpose(x0_shuf_patches_0, perm=[1, 2, 3, 0])
					x0_shuf_patches_1=tf.transpose(x0_shuf_patches_1, perm=[1, 2, 3, 0])
					x0_shuf_patches_2=tf.transpose(x0_shuf_patches_2, perm=[1, 2, 3, 0])
					x1_shuf_patches=tf.transpose(x1_patches, perm=[3, 0, 1, 2])
					x1_shuf_patches_0 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_0, tf.int32))
					x1_shuf_patches_1 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_1, tf.int32))
					x1_shuf_patches_2 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_2, tf.int32))
					x1_shuf_patches_0=tf.transpose(x1_shuf_patches_0, perm=[1, 2, 3, 0])
					x1_shuf_patches_1=tf.transpose(x1_shuf_patches_1, perm=[1, 2, 3, 0])
					x1_shuf_patches_2=tf.transpose(x1_shuf_patches_2, perm=[1, 2, 3, 0])
					x0_pos=(1+binarize(x0_patches))/2*abs(x0_patches)
					x0_neg=(1-binarize(x0_patches))/2*abs(x0_patches)
					x1_pos=(1+binarize(x1_patches))/2*abs(x1_patches)
					x1_neg=(1-binarize(x1_patches))/2*abs(x1_patches)
					x0s0_pos=(1+binarize(x0_shuf_patches_0))/2
					x0s0_neg=(1-binarize(x0_shuf_patches_0))/2
					x1s0_pos=(1+binarize(x1_shuf_patches_0))/2
					x1s0_neg=(1-binarize(x1_shuf_patches_0))/2
					x0s1_pos=(1+binarize(x0_shuf_patches_1))/2
					x0s1_neg=(1-binarize(x0_shuf_patches_1))/2
					x1s1_pos=(1+binarize(x1_shuf_patches_1))/2
					x1s1_neg=(1-binarize(x1_shuf_patches_1))/2
					x0s2_pos=(1+binarize(x0_shuf_patches_2))/2
					x0s2_neg=(1-binarize(x0_shuf_patches_2))/2
					x1s2_pos=(1+binarize(x1_shuf_patches_2))/2
					x1s2_neg=(1-binarize(x1_shuf_patches_2))/2

					self.out=         K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_w1, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_w2, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_w3, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_w4, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_w5, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_w6, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_w7, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_w8, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_w9, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_w10, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_w11, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_w12, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_w13, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_w14, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_w15, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_w16, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_w17, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_w18, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_w19, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_w20, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_w21, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_w22, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_w23, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_w24, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_w25, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_w26, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_w27, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_w28, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_w29, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_w30, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_w31, [-1, self.nfilters])*self.pruning_mask)
					self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_w32, [-1, self.nfilters])*self.pruning_mask)

					#self.out=K.conv2d(x_pos[0,:,:,:,:]*xs_pos[0,:,:,:,:], kernel=self.clamped_w1, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_pos[0,:,:,:,:]*xs_neg[0,:,:,:,:], kernel=self.clamped_w2, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*xs_pos[0,:,:,:,:], kernel=self.clamped_w3, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*xs_neg[0,:,:,:,:], kernel=self.clamped_w4, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_pos[1,:,:,:,:]*xs_pos[1,:,:,:,:], kernel=self.clamped_w5, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_pos[1,:,:,:,:]*xs_neg[1,:,:,:,:], kernel=self.clamped_w6, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_neg[1,:,:,:,:]*xs_pos[1,:,:,:,:], kernel=self.clamped_w7, padding=self.padding,strides=self.strides )
					#self.out=self.out+K.conv2d(x_neg[1,:,:,:,:]*xs_neg[1,:,:,:,:], kernel=self.clamped_w8, padding=self.padding,strides=self.strides )

				else:
					x_expanded=0
					for l in range(self.levels):
						x_in=x[l,:,:,:,:]
						x_expanded=x_expanded+x_in
					self.out=K.conv2d(x_expanded, kernel=self.clamped_w*tf.reshape(self.pruning_mask, [self.k, self.k, self.ch_in, self.nfilters]), padding=self.padding,strides=self.strides )

		self.output_dim=self.out.get_shape()
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

class binary_dense(Layer):
	def __init__(self,n_in,n_out,levels=1,pruning_prob=0,first_layer=False,LUT=True,BINARY=True,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		self.levels=levels
		self.LUT=LUT
		self.BINARY=BINARY
		self.first_layer=first_layer
		super(binary_dense,self).__init__(**kwargs)
	def build(self, input_shape):
		self.rand_map_0 = self.add_weight(name='rand_map_0', 
			shape=(self.n_in, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
			trainable=False) # Randomisation map for subsequent input connections
		self.rand_map_1 = self.add_weight(name='rand_map_1', 
			shape=(self.n_in, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
			trainable=False) # Randomisation map for subsequent input connections
		self.rand_map_2 = self.add_weight(name='rand_map_2', 
			shape=(self.n_in, 1),
			initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
			trainable=False) # Randomisation map for subsequent input connections

		stdv=1/np.sqrt(self.n_in)
		self.gamma=K.variable(1.0)
		if self.levels==1 or self.first_layer==True:
			w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
			self.w=K.variable(w)
			self.trainable_weights=[self.w,self.gamma]
		elif self.levels==2:
			if self.LUT==True:

				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w9 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w10 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w11 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w12 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w13 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w14 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w15 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w16 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w17 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w18 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w19 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w20 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w21 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w22 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w23 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w24 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w25 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w26 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w27 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w28 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w29 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w30 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w31 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w32 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)

				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)
				self.w9=K.variable(w9)
				self.w10=K.variable(w10)
				self.w11=K.variable(w11)
				self.w12=K.variable(w12)
				self.w13=K.variable(w13)
				self.w14=K.variable(w14)
				self.w15=K.variable(w15)
				self.w16=K.variable(w16)
				self.w17=K.variable(w17)
				self.w18=K.variable(w18)
				self.w19=K.variable(w19)
				self.w20=K.variable(w20)
				self.w21=K.variable(w21)
				self.w22=K.variable(w22)
				self.w23=K.variable(w23)
				self.w24=K.variable(w24)
				self.w25=K.variable(w25)
				self.w26=K.variable(w26)
				self.w27=K.variable(w27)
				self.w28=K.variable(w28)
				self.w29=K.variable(w29)
				self.w30=K.variable(w30)
				self.w31=K.variable(w31)
				self.w32=K.variable(w32)

				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.w9,self.w10,self.w11,self.w12,self.w13,self.w14,self.w15,self.w16,self.w17,self.w18,self.w19,self.w20,self.w21,self.w22,self.w23,self.w24,self.w25,self.w26,self.w27,self.w28,self.w29,self.w30,self.w31,self.w32,self.gamma]



			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]
		elif self.levels==3:
			if self.LUT==True:
				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)

				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.gamma]
			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]

		self.pruning_mask = self.add_weight(name='pruning_mask',
			shape=(self.n_in,self.n_out),
			initializer=keras.initializers.Constant(value=np.ones((self.n_in,self.n_out))),
			trainable=False) # LUT pruning based on whether inputs get repeated


#		elif self.levels==2:#train baseline without resid gamma scaling
#			w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
#			self.w=K.variable(w)
#			self.trainable_weights=[self.w,self.gamma]


	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)
		if self.levels==1 or self.first_layer==True:
			if self.BINARY==False:
				self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
			else:
				self.clamped_w=constraint_gamma*binarize(self.w)
			self.out=K.dot(x,self.clamped_w)
		elif self.levels==2:
			if self.LUT==True:
				if self.BINARY==False:
					self.clamped_w1=constraint_gamma*K.clip(self.w1,-1,1)
					self.clamped_w2=constraint_gamma*K.clip(self.w2,-1,1)
					self.clamped_w3=constraint_gamma*K.clip(self.w3,-1,1)
					self.clamped_w4=constraint_gamma*K.clip(self.w4,-1,1)
					self.clamped_w5=constraint_gamma*K.clip(self.w5,-1,1)
					self.clamped_w6=constraint_gamma*K.clip(self.w6,-1,1)
					self.clamped_w7=constraint_gamma*K.clip(self.w7,-1,1)
					self.clamped_w8=constraint_gamma*K.clip(self.w8,-1,1)
					self.clamped_w9=constraint_gamma*K.clip(self.w9,-1,1)
					self.clamped_w10=constraint_gamma*K.clip(self.w10,-1,1)
					self.clamped_w11=constraint_gamma*K.clip(self.w11,-1,1)
					self.clamped_w12=constraint_gamma*K.clip(self.w12,-1,1)
					self.clamped_w13=constraint_gamma*K.clip(self.w13,-1,1)
					self.clamped_w14=constraint_gamma*K.clip(self.w14,-1,1)
					self.clamped_w15=constraint_gamma*K.clip(self.w15,-1,1)
					self.clamped_w16=constraint_gamma*K.clip(self.w16,-1,1)
					self.clamped_w17=constraint_gamma*K.clip(self.w17,-1,1)
					self.clamped_w18=constraint_gamma*K.clip(self.w18,-1,1)
					self.clamped_w19=constraint_gamma*K.clip(self.w19,-1,1)
					self.clamped_w20=constraint_gamma*K.clip(self.w20,-1,1)
					self.clamped_w21=constraint_gamma*K.clip(self.w21,-1,1)
					self.clamped_w22=constraint_gamma*K.clip(self.w22,-1,1)
					self.clamped_w23=constraint_gamma*K.clip(self.w23,-1,1)
					self.clamped_w24=constraint_gamma*K.clip(self.w24,-1,1)
					self.clamped_w25=constraint_gamma*K.clip(self.w25,-1,1)
					self.clamped_w26=constraint_gamma*K.clip(self.w26,-1,1)
					self.clamped_w27=constraint_gamma*K.clip(self.w27,-1,1)
					self.clamped_w28=constraint_gamma*K.clip(self.w28,-1,1)
					self.clamped_w29=constraint_gamma*K.clip(self.w29,-1,1)
					self.clamped_w30=constraint_gamma*K.clip(self.w30,-1,1)
					self.clamped_w31=constraint_gamma*K.clip(self.w31,-1,1)
					self.clamped_w32=constraint_gamma*K.clip(self.w32,-1,1)

				else:

					self.clamped_w1=constraint_gamma*binarize(self.w1)
					self.clamped_w2=constraint_gamma*binarize(self.w2)
					self.clamped_w3=constraint_gamma*binarize(self.w3)
					self.clamped_w4=constraint_gamma*binarize(self.w4)
					self.clamped_w5=constraint_gamma*binarize(self.w5)
					self.clamped_w6=constraint_gamma*binarize(self.w6)
					self.clamped_w7=constraint_gamma*binarize(self.w7)
					self.clamped_w8=constraint_gamma*binarize(self.w8)
					self.clamped_w9=constraint_gamma*binarize(self.w9)
					self.clamped_w10=constraint_gamma*binarize(self.w10)
					self.clamped_w11=constraint_gamma*binarize(self.w11)
					self.clamped_w12=constraint_gamma*binarize(self.w12)
					self.clamped_w13=constraint_gamma*binarize(self.w13)
					self.clamped_w14=constraint_gamma*binarize(self.w14)
					self.clamped_w15=constraint_gamma*binarize(self.w15)
					self.clamped_w16=constraint_gamma*binarize(self.w16)
					self.clamped_w17=constraint_gamma*binarize(self.w17)
					self.clamped_w18=constraint_gamma*binarize(self.w18)
					self.clamped_w19=constraint_gamma*binarize(self.w19)
					self.clamped_w20=constraint_gamma*binarize(self.w20)
					self.clamped_w21=constraint_gamma*binarize(self.w21)
					self.clamped_w22=constraint_gamma*binarize(self.w22)
					self.clamped_w23=constraint_gamma*binarize(self.w23)
					self.clamped_w24=constraint_gamma*binarize(self.w24)
					self.clamped_w25=constraint_gamma*binarize(self.w25)
					self.clamped_w26=constraint_gamma*binarize(self.w26)
					self.clamped_w27=constraint_gamma*binarize(self.w27)
					self.clamped_w28=constraint_gamma*binarize(self.w28)
					self.clamped_w29=constraint_gamma*binarize(self.w29)
					self.clamped_w30=constraint_gamma*binarize(self.w30)
					self.clamped_w31=constraint_gamma*binarize(self.w31)
					self.clamped_w32=constraint_gamma*binarize(self.w32)



				# Special hack for randomising the subsequent input connections: tensorflow does not support advanced matrix indexing
				shuf_x=tf.transpose(x, perm=[2, 0, 1])
				shuf_x_0 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_0, tf.int32))
				shuf_x_1 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_1, tf.int32))
				shuf_x_2 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_2, tf.int32))
				shuf_x_0=tf.transpose(shuf_x_0, perm=[1, 2, 0])
				shuf_x_1=tf.transpose(shuf_x_1, perm=[1, 2, 0])
				shuf_x_2=tf.transpose(shuf_x_2, perm=[1, 2, 0])
				x_pos=(1+binarize(x))/2*abs(x)
				x_neg=(1-binarize(x))/2*abs(x)
				xs0_pos=(1+binarize(shuf_x_0))/2
				xs0_neg=(1-binarize(shuf_x_0))/2
				xs1_pos=(1+binarize(shuf_x_1))/2
				xs1_neg=(1-binarize(shuf_x_1))/2
				xs2_pos=(1+binarize(shuf_x_2))/2
				xs2_neg=(1-binarize(shuf_x_2))/2

				self.out=         K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_w1*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_w2*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_w3*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_w4*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_w5*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_w6*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_w7*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_w8*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_w9*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_w10*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_w11*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_w12*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_w13*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_w14*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_w15*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_w16*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_w17*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_w18*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_w19*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_w20*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_w21*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_w22*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_w23*self.pruning_mask)
				self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_w24*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_w25*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_w26*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_w27*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_w28*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_w29*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_w30*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_w31*self.pruning_mask)
				self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_w32*self.pruning_mask)

			else:
				x_expanded=0
#				self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
				self.clamped_w=constraint_gamma*binarize(self.w)
				for l in range(self.levels):
					x_expanded=x_expanded+x[l,:,:]
				self.out=K.dot(x_expanded,self.clamped_w*self.pruning_mask)
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.n_out)
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.n_out)


class my_flat(Layer):
	def __init__(self,**kwargs):
		super(my_flat,self).__init__(**kwargs)
	def build(self, input_shape):
		return

	def call(self, x, mask=None):
		self.out=tf.reshape(x,[-1,np.prod(x.get_shape().as_list()[1:])])
		return self.out
	def  compute_output_shape(self,input_shape):
		shpe=(input_shape[0],int(np.prod(input_shape[1:])))
		return shpe
