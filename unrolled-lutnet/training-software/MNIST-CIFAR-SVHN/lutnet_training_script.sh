#!/bin/bash

echo -e "Please make sure you have updated the folding factors in 'init_prune.py'. Please enter test id: "
read id
echo -e "Please enter dataset (CIFAR-10/SVHN/MNIST):"
read dataset
echo "Start test $id"

if [ $dataset == 'MNIST' ]
then
	trainEpochs=50
	retrainEpochs=10
elif [ $dataset == 'CIFAR-10' ] || [ $dataset == 'SVHN' ]
then
	trainEpochs=200
	retrainEpochs=50
else
	echo -e "Please make sure that the dataset is one of (CIFAR-10/SVHN/MNIST)."
	exit
fi

cd models/${dataset}/scripts
python bnn_pruning.py
cp pretrained_pruned.h5 ../pretrained_pruned.h5
cd ../../..
python Binary.py ${dataset} True False True False True True True ${retrainEpochs} > output.txt
mkdir -p models/${dataset}/pruned_bnn
cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_bnn/pruned_bnn_${id}.h5
cp output.txt models/${dataset}/pruned_bnn/pruned_bnn_${id}.txt

cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_pruned.h5
cd models/${dataset}/scripts
python lutnet_init.py
cp pretrained_bin.h5 ../pretrained_bin.h5
cd ../../..
python Binary.py ${dataset} True False False True True False True ${trainEpochs} > output.txt
mkdir -p models/${dataset}/pruned_lutnet
cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_lutnet/pruned_lutnet_${id}_BIN.h5
cp output.txt models/${dataset}/pruned_lutnet/pruned_lutnet_${id}_BIN.txt
