# LUTNet-Learning-FPGA-Configurations-for-Efficient-Neural-Network-Inference

## Training LUTNet

### Prerequisites

For training LUTNet, you should have the following packages installed:
* Keras (v2)
* TensorFlow

### Step 1: Train BNN (ReBNet) From Scratch

```
cd training-software/MNIST-CIFAR-SVHN/
bash bnn_regularised_training.sh
bash dummy_generation.sh
```

Select the dataset (MNIST/CIFAR-10/SVHN) when prompted.

### Step 2: Logic Expansion (Retrain Using LUTNet Architecture)

Open `models/${dataset}/scripts/bnn_pruning.py` and edit each layer's pruning threshold. Below is an example with LFC model classifying MNIST (-1 means no pruning).

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

Select the test id (an index number to distinguish between multiple test outputs) and the dataset when prompted. After training finishes, the pretrained network and accuracy results for the intermediate BNN and final LUTNet can be found in `models/${dataset}/pruned_bnn` and `models/${dataset}/pruned_lutnet`, respectively.
