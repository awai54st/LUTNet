#!/bin/bash

echo -e "Please enter the test id of pretrained bnn that you'd like to retrieve: "
read id

cp ../../../training-software/MNIST-CIFAR-SVHN/models/MNIST/pruned_bnn/pruned_bnn_${id}.h5 pretrained_network_reb_tm_mnist.h5
python h52header_reb_tm_mnist_spase.py
cp ../codegen_output/weights.h ../../src/network/MNIST/hw/weights_reb.h 
