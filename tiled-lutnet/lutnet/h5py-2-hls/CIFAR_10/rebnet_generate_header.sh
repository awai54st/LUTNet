#!/bin/bash

echo -e "Please enter the test id of pretrained bnn that you'd like to retrieve: "
read id

cp ../../../training-software/MNIST-CIFAR-SVHN/models/CIFAR-10/pruned_bnn/pruned_bnn_${id}.h5 pretrained_network_reb_tm.h5
python h52header_reb_tm_spase.py
cp ../codegen_output/weights.h ../../src/network/CIFAR10/hw/weights_reb.h
