#!/bin/bash

echo -e "Please enter the test id of pretrained lutnet that you'd like to retrieve: "
read id

cp ../../../training-software/MNIST-CIFAR-SVHN/models/MNIST/pruned_lutnet/pruned_lutnet_${id}_BIN.h5 pretrained_network_51lut_tm_mnist.h5
python h52header_51lut_tm_mnist_spase.py
cp ../codegen_output/LUTARRAY.v ../codegen_output/LUTARRAY_1.v ../codegen_output/LUTARRAY_2.v ../codegen_output/LUTARRAY_3.v ../../src/network/LUTNET_c6/sol1/syn/verilog/
cp ../codegen_output/weights.h ../../src/network/MNIST/hw/weights.h

