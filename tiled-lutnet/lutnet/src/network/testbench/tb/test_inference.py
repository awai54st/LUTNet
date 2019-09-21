import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import keras
from keras.datasets import cifar10,mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
from binarization_utils import *
from model_architectures import get_model

dataset='CIFAR-10'

EVALUATE = False
DEBUG = True

from PIL import Image

if DEBUG:

	test_image = Image.open('cat1.png')
	#test_image = Image.open('airplane1.png')
	test_image.thumbnail((32, 32), Image.ANTIALIAS)
	background = Image.new('RGB', (32, 32), (255, 255, 255))
	background.paste(
	    test_image, (int((32 - test_image.size[0]) / 2), int((32 - test_image.size[1]) / 2))
	)
	# We write the image into the format used in the Cifar-10 dataset for code compatibility 
	img = (np.array(background))
	img = np.reshape(img, (-1, 32, 32, 3))
	X_test = img
	Y_test = np.reshape(np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), (-1,10))
	X_test=X_test.astype(np.float32)
	X_test /= 255
	X_test=2*X_test-1

	print('X_test shape:', X_test.shape)
	print('y_test shape:', Y_test.shape)
	
	resid_levels = 2

	batch_norm_eps=1e-4
	batch_norm_alpha=0.1#(this is same as momentum)

	model=Sequential()
	model.add(binary_conv(pruning_prob=0.1,nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,Prune=True))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))
	model.add(binary_conv(levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=64,k=3,padding='valid',LUT=True,Prune=True))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
        model.add(Residual_sign(levels=resid_levels))
        model.add(binary_conv(levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=64,k=3,padding='valid',LUT=True,Prune=True))
        model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
        model.add(Residual_sign(levels=resid_levels))
        model.add(binary_conv(levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=128,k=3,padding='valid',LUT=True,Prune=True))
        model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
        model.add(Residual_sign(levels=resid_levels))

        model.add(binary_conv(levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=128,k=3,padding='valid',LUT=True,Prune=True))
        model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
        model.add(Residual_sign(levels=resid_levels))
        model.add(binary_conv(levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=256,k=3,padding='valid',LUT=True,Prune=True))
        model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))

        #model.add(my_flat())

        #model.add(Residual_sign(levels=resid_levels))
        #model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=True,Prune=True))
        #model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
        #model.add(Residual_sign(levels=resid_levels))
        #model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=True,Prune=True))
        #model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	weights_path='pretrained_network.h5'
	model.load_weights(weights_path, by_name=True)
	prediction = model.predict(X_test)
	print(np.shape(prediction))
	print(prediction[0][0][0])
	#print(prediction.transpose(0, 3, 1, 2)[0][0])

if EVALUATE:
	if dataset=="MNIST":
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		# convert class vectors to binary class matrices
		X_train = X_train.reshape(-1,784)
		X_test = X_test.reshape(-1,784)
		use_generator=False
	elif dataset=="CIFAR-10":
		use_generator=True
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	elif dataset=="SVHN":
		use_generator=True
		(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN].")
	
	X_train=X_train.astype(np.float32)
	X_test=X_test.astype(np.float32)
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)
	X_train /= 255
	X_test /= 255
	X_train=2*X_train-1
	X_test=2*X_test-1
	
	print('X_test shape:', X_test.shape)
	print('y_test shape:', Y_test.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	resid_levels = 2	

	weights_path='baseline_reg.h5'
	model=get_model(dataset,resid_levels)
	model.load_weights(weights_path)
	opt = keras.optimizers.Adam()
	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	#model.summary()
	score=model.evaluate(X_test,Y_test,verbose=0)
	print "with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1])

