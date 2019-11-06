# LUTNet: Rethinking Inference in FPGA Soft Logic

## Repo organisation

The repo contains two versions of LUTNet.

* __Unrolled LUTNet__: Operators in convolutional layers are mapped to FPGA resources with one-to-one LUT binding. No BRAM is consumed for weight storage as weights are hardened in LUT configuration masks. Details can be found in our paper _LUTNet: Rethinking Inference in FPGA Soft Logic_.
* __Tiled LUTNet__: Operators are tiled and reused, trading off area efficiency for resource savings. BRAMs are consumed for weight storage.

## Prerequisites

For training LUTNet, you should have the following packages installed:
* Keras (v2)
* TensorFlow

For hardware synthesis, we developed and tested the project with Vivado (+ HLS) 2016.3. 
Newer versions of Vivado HLS do not work with our project. 
In newer versions of Vivado HLS, loop unrolling factors are limited, reducing the area-efficiency advantage of LUTNet.

## Results

<table>
  <tr>
    <td>Dataset</td>
    <td>Tiling</td>
    <td>Top-1 Accuracy (%)</td>
    <td>LUT</td>
    <td>FPS</td>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>Fully unrolled</td>
    <td>98.01</td>
    <td>58192</td>
    <td>200M</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>Largest conv layer fully unrolled</td>
    <td>96.20</td>
    <td>154814</td>
    <td>200M (Target layer only)</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>Tiled</td>
    <td>96.42</td>
    <td>361528</td>
    <td>10.2k</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>Largest conv layer fully unrolled</td>
    <td>84.21</td>
    <td>246044</td>
    <td>200M (Target layer only)</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>Tiled</td>
    <td>84.80</td>
    <td>106776</td>
    <td>10.2k</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>Largest conv layer fully unrolled</td>
    <td>41.45</td>
    <td>496160</td>
    <td>5.56M (Target layer only)</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>Tiled</td>
    <td>41.28</td>
    <td>482903</td>
    <td>260</td>
  </tr>
</table>

## Citation

If you make use of this code, please acknowledge us by citing our [conference paper](https://arxiv.org/abs/1904.00938).

    @inproceedings{lutnet,
		author={Wang, Erwei and Davis, James J. and Cheung, Peter Y. K. and Constantinides, George A.},
		title={{LUTNet}: Rethinking Inference in {FPGA} Soft Logic},
		booktitle={IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
		year = {2019}
    }

## References

### 1. ReBNet

    @inproceedings{rebnet,
		author = {Mohammad Ghasemzadeh and Mohammad Samragh and Farinaz Koushanfar},
		title = {{ReBNet}: Residual Binarized Neural Network},
		booktitle = {IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
		year = {2018}
    }

### 2. FINN

    @inproceedings{finn,
		author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
		title = {{FINN}: A Framework for Fast, Scalable Binarized Neural Network Inference},
		booktitle = {ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA)},
		year = {2017}
    }
