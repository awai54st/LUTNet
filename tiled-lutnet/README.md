# LUTNet: Learning FPGA Configurations for Efficient Neural Network Inference

## Training LUTNet

The training software of LUTNet uses a train-prune-retrain workflow. 
Here, we use the LFC model with MNIST dataset as an example.

### Step 1: Train BNN (ReBNet) From Scratch

```
cd training-software/MNIST-CIFAR-SVHN/
bash bnn_regularised_training.sh
bash dummy_generation.sh
```

Select the dataset (MNIST/CIFAR-10/SVHN) when prompted.

### Step 2: BNN Fine-grained Pruning + Logic Expansion (Retraining with LUTNet Architecture)

Open `models/${dataset}/scripts/bnn_pruning.py` and edit each layer's pruning threshold. Below is an example with LFC model classifying MNIST (-1 means no pruning), where a higher threshold corresponds to more aggressive pruning.

```
p_d1=-1
p_d2=0.78
p_d3=0.78
p_d4=0.78
p_d5=-1
```
Then, execute the LUTNet retraining script.

```
bash lutnet_training_script.sh
```

Select the test id (an identifier that distinguishes among multiple test outputs) and the dataset when prompted. After training finishes, the pretrained network and accuracy results for the intermediate BNN and final LUTNet can be found in `models/${dataset}/pruned_bnn` and `models/${dataset}/pruned_lutnet`, respectively.

## Mapping a Trained LUTNet on an FPGA

The pretrained LUTNet (in .h5 format) is converted into RTL (verilog) and then synthesised into an FPGA bitstream.

### Step 1: Convert a Pretrained LUTNet into C Headers and Verilog Source Codes

```
cd lutnet/h5py-2-hls/MNIST
bash 51lutnet_mnist_generate_lutarray.sh
```
Enter the test id of the pretrained network (that you'd like to implement) when prompted.
The script generates two sets of source codes: LUT arrays in verilog format and other parameters (bn thresholds, scaling factors etc.) in C header format.
The file copy of LUT array verilog files may fail because the destination folder `/lutnet/src/network/LUTNET_c6/` does not exist yet.
This is normal as we will run this script again after HLS, and the folder will then exist.

### Step 2: HLS

```
cd ../../lutnet/src/network/
bash lutnet_synthesis_script_part1.sh
```
Wait for HLS to finish. 
For CNV it could take up to a day.
The HLS output directory is `LUTNET_c6/`.
Inside, the LUT arrays are synthesised as place holder modules which contains no meaningful logic.
We now replace those place holders with LUT array verilogs that we have generated in Step 1.

IMPORTANT: After HLS finishes, open `LUTNET_c6/sol/syn/verilog/DoCompute.v` and scroll all the way down to around line 530 (may vary randomly) where the FIFO modules are instantiated.
For reasons unknown to me (I think it is a bug with HLS) only some of these modules were created (again, randomly) whereas others do not exist.
My workaround is to check which modules are created and use them to replace the missing ones -- they are functionally identical.

Then, go back to `lutnet/h5py-2-hls/MNIST` and `bash 51lutnet_mnist_generate_lutarray.sh` again.
This time the file copies should be successful, and the LUT array place holders are replaced.

### Step 3: Vivado Synthesis.

```
bash lutnet_synthesis_script_part2.sh
```

The final step, bitstream generation, will fail as the pin assignment is not complete.
But you can still obtain the post-placement utilisation and power consumption reports under `src/network/vivado_output`.

## Custom Models

For advices on how to make changes to the models please see [ADVANCED.md](ADVANCED.md).
