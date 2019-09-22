
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.python.framework import ops
from binarization_utils import *

batch_norm_eps=1e-4
batch_norm_alpha=0.1#(this is same as momentum)

def get_model(dataset,resid_levels,LUT,BINARY,trainable_means):
	if dataset=='MNIST':
		model=Sequential()
		model.add(binary_dense(levels=resid_levels,n_in=784,n_out=256,input_shape=[784],first_layer=True,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=LUT,BINARY=BINARY,TM=8,TN=10))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))

	
	elif dataset=="CIFAR-10" or dataset=="SVHN":
		model=Sequential()
		model.add(binary_conv(pruning_prob=0.1,nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY,TM=1,TN=2))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_conv(levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=64,k=3,padding='valid',LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))

		model.add(binary_conv(levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=64,k=3,padding='valid',LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_conv(levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=128,k=3,padding='valid',LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))

		model.add(binary_conv(levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=128,k=3,padding='valid',LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_conv(levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=256,k=3,padding='valid',LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))

		model.add(my_flat())

		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=LUT,BINARY=BINARY,TM=8,TN=8))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.5,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=LUT,BINARY=BINARY,TM=8,TN=10))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))
	else:
		raise("dataset should be one of the following list: [MNIST, CIFAR-10, SVHN, Imagenet].")
	return model
