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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0, '..')
from binarization_utils import *
from model_architectures import get_model

dataset = sys.argv[1]
Train = sys.argv[2] == 'True'
REG = sys.argv[3] == 'True'
Retrain = sys.argv[4] == 'True'
LUT = sys.argv[5] == 'True'
BINARY = sys.argv[6] == 'True'
trainable_means = sys.argv[7] == 'True'
Evaluate = sys.argv[8] == 'True'
epochs = int(sys.argv[9])

batch_size=100

print('Dataset is ', dataset)
print('Train is ', Train)
print('REG is ', REG)
print('Retrain is ', Retrain)
print('LUT is ', LUT)
print('BINARY is ', BINARY)
print('trainable_means is ', trainable_means)
print('Evaluate is ', Evaluate)

def l2_reg(weight_matrix):
	return 5e-7 * K.sqrt(K.sum(K.abs(weight_matrix)**2))

def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1

	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1

	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1

	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)

	return (X_train,y_train),(X_test,y_test)

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


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')





if Train:
	if not(os.path.exists('models')):
		os.mkdir('models')
	if not(os.path.exists('models/'+dataset)):
		os.mkdir('models/'+dataset)
	for resid_levels in range(2,3): #range(1,4):
		print 'training with', resid_levels,'levels'
		sess=K.get_session()
		model=get_model(dataset,resid_levels,LUT,BINARY,trainable_means)
		#model.summary()

		#gather all binary dense and binary convolution layers:
		binary_layers=[]
		for l in model.layers:
			if isinstance(l,binary_dense) or isinstance(l,binary_conv):
				binary_layers.append(l)

		#gather all residual binary activation layers:
		resid_bin_layers=[]
		for l in model.layers:
			if isinstance(l,Residual_sign):
				resid_bin_layers.append(l)
		lr=0.01
		decay=1e-6

		if Retrain:
			weights_path='models/'+dataset+'/pretrained_pruned.h5'
			model.load_weights(weights_path)
		elif LUT:
			weights_path='models/'+dataset+'/pretrained_bin.h5'
			model.load_weights(weights_path)
		else:
			vars   = tf.trainable_variables() 
			if dataset == 'CIFAR-10' or dataset == 'SVHN':
				lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
					if 'binary_conv_6' in v.name and 'Variable_' in v.name ]) * 5e-7 # reg applied onto conv_6 only
			elif dataset == 'MNIST':
				lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
					if ('binary_dense_' in v.name and 'binary_dense_1' not in v.name) and 'Variable_' in v.name ]) * 5e-7 # reg applied onto conv_6 only
			model.add_loss( lossL2 )

		opt = keras.optimizers.Adam(lr=lr,decay=decay)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
		if use_generator:
			if dataset=="CIFAR-10":
				horizontal_flip=True
			if dataset=="SVHN":
				horizontal_flip=False
			datagen = ImageDataGenerator(
				width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip)  # randomly flip images
			if keras.__version__[0]=='2':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
				nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0], 
				nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[cback])

		else:
			if keras.__version__[0]=='2':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[cback])
		dic={'hard':history.history}
		foo=open('models/'+dataset+'/history_'+str(resid_levels)+'_residuals.pkl','wb')
		pickle.dump(dic,foo)
		foo.close()

if Evaluate:
	for resid_levels in range(2,3):
		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels,LUT,BINARY,trainable_means)
		model.load_weights(weights_path)
		opt = keras.optimizers.Adam()
		model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.summary()
		score=model.evaluate(X_test,Y_test,verbose=0)
		print "with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1])


