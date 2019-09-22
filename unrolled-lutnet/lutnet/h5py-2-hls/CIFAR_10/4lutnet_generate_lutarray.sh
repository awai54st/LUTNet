#!/bin/bash

echo -e "Please enter the test id of pretrained lutnet that you'd like to retrieve: "
read id

mkdir -p ../codegen_output
cp ../../../training-software/MNIST-CIFAR-SVHN/models/CIFAR-10/pruned_lutnet/pruned_lutnet_${id}_BIN.h5 pretrained_network_4lut.h5
python h52header_4lut_spase.py
cp ../codegen_output/LUTARRAY*.v ../../src/network/LUTNET_c6/sol1/syn/verilog/
cp ../codegen_output/weights.h ../../src/network/CIFAR10/hw/weights.h

