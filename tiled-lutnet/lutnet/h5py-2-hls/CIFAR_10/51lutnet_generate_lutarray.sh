#!/bin/bash

echo -e "Please enter the test id of pretrained lutnet that you'd like to retrieve: "
read id

cp ../../../training-software/MNIST-CIFAR-SVHN/models/CIFAR-10/pruned_lutnet/pruned_lutnet_${id}_BIN.h5 pretrained_network_51lut_tm.h5
python h52header_51lut_tm_spase.py
mkdir -p ../codegen_output
cp ../codegen_output/LUTARRAY.v ../codegen_output/LUTARRAY_1.v ../codegen_output/LUTARRAY_2.v ../codegen_output/LUTARRAY_3.v ../codegen_output/LUTARRAY_4.v ../codegen_output/LUTARRAY_5.v ../codegen_output/LUTARRAY_6.v ../codegen_output/LUTARRAY_7.v ../../src/network/LUTNET_c6/sol1/syn/verilog/
cp ../codegen_output/weights.h ../../src/network/CIFAR10/hw/weights.h

