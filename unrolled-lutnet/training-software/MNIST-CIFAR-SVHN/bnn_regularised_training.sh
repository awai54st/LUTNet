#!/bin/bash

echo -e "Please enter dataset (CIFAR-10/SVHN/MNIST):"
read dataset
echo "Start training bnn from scratch."

if [ $dataset == 'MNIST' ]
then
        trainEpochs=50
elif [ $dataset == 'CIFAR-10' ] || [ $dataset == 'SVHN' ]
then
        trainEpochs=200
else
        echo -e "Please make sure that the dataset is one of (CIFAR-10/SVHN/MNIST)."
        exit
fi

python Binary.py ${dataset} True True False False False True False ${trainEpochs} > output.txt

cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_reg.h5
cp output.txt models/${dataset}/scripts/baseline_reg.txt

echo -e "Finished training bnn from scratch. For LUTNet please run lutnet_training_script.sh."

