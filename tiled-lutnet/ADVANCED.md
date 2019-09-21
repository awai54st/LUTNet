# Advices on Modifying the Project

Generally, if you make architectural modifications to training software, please remember to change the HLS parameters accordingly.

## For Each New HLS Run

For reasons unknown, in each HLS run, the HLS tool will randomise the order of CONV/FC layers, meaning that LUTARRAY_1.v is almost never going to be the first layer.
Therefore, after the HLS run, we need to always look at the synthesised varilog and keep track of the order of layers.
This is done by looking at the layer id inside the generated source codes `tiled-lutnet/lutnet/src/network/LUTNET_c6/sol/syn/verilog/LUTARRAY*.v`.
For each file, look for a line that looks something like this.
```
parameter    ap_const_lv36_1 = 36'b1;
```
In this example, it means this source file LUTARRAY*.v corresponds to the first LUTNet layer.
After you record down the order of the layers, update them at (around) line 624 in `tiled-lutnet/lutnet/h5py-2-hls/${dataset}/h52header_51lut_tm_mnist_spase.py` before proceeding to generate the LUT array verilog files.

## Change Tiling Factors

The following source files should be changed accordingly.
```
tiled-lutnet/training-software/model_architectures.py
tiled-lutnet/training-software/MNIST-CIFAR-SVHN/Binary.py
tiled-lutnet/training-software/MNIST-CIFAR-SVHN/models/${dataset}/scripts/bnn_pruning.py
tiled-lutnet/lutnet/h5py-2-hls/${dataset}/h52header_51lut_tm_mnist_spase.py
tiled-lutnet/lutnet/src/network/MNIST/hw/config.h
```

IMPORTANT: Tiling factor of "1" (i.e. full unrolling) doesn't work with this project.
HLS will generate a completely sequential implementation instead (a complete opposite to full unrolling).
This is due to the coding style of HLS C source codes in this project.
For fully unrolled LUTNet/ReBNet layers, please go to `LUTNet/unrolled-lutnet`.
Mix and match of tiled and fully unrolled LUTNet layers is possible.

## Reproduce ReBNet Results

This repository also includes the training and implementation source codes of ReBNet.
Implementing ReBNet takes similar steps as LUTNet, except LUT array replacement is not needed -- all BNN weights are stored in the generated C header file `weight.h`.

## Change Microarchitecture

The default microarchitecture is (5,1)-LUTNet.
The source codes for other microarchitectures are included in this repository.

## LUTNet-ReBNet Hybrids

Mixing and matching LUTNet and ReBNet layers are supported.
The following source codes should be modified.
```
tiled-lutnet/training-software/model_architectures.py (LUT=LUT for LUTNet and LUT=False for ReBNet)
tiled-lutnet/lutnet/h5py-2-hls/MNIST/h52header_51lut_tm_mnist_spase.py
tiled-lutnet/lutnet/src/network/MNIST/hw/top.cpp
```
