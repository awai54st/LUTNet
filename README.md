# LUTNet: Learning FPGA Configurations for Efficient Neural Network Inference

## Repo organization

The repo contains two versions of LUTNet.

* __Fully unrolled LUTNet__: Operators in a convolution layer are mapped to FPGA with one-to-one LUT binding. No BRAM is consumed in dot product as weights are all hardened in LUT binary masks. Details can be found in our paper _LUTNet: Rethinking Inference in FPGA Soft Logic_.
* __Tiled LUTNet__: Operators are tiled and reused. Trades off area efficiency with resource savings. BRAMs are consumed in dot product. Details can be found in our paper _LUTNet: Learning FPGA Configurations forHighly Efficient Neural Network Inference_.

## Prerequisites

For training LUTNet, you should have the following packages installed:
* Keras (v2)
* TensorFlow

For hardware synthesis, we developed and tested the project with Vivado (+ HLS) 2016.3. 
Newer versions of Vivado HLS do not work with our project. 
In newer versions of HLS the loop unrolling factor is limited to a small number which limits the area-efficiency advantage of LUTNet.

## Citation
If you find LUTNet useful, please cite our [FCCM'19 conference paper](https://arxiv.org/abs/1904.00938).

    @inproceedings{lutnet,
    author={Wang, Erwei and Davis, James J and Cheung, Peter YK and Constantinides, George A},
    title={LUTNet: Rethinking Inference in FPGA Soft Logic},
    booktitle={2019 IEEE 27th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
    pages={26-34},
    doi={10.1109/FCCM.2019.00014},
    month={April},
    year = {2019}
    }

## References

### 1. ReBNet

If you find ReBNet useful, please cite the <a href="https://arxiv.org/abs/1711.01243" target="_blank">ReBNet paper</a>:

    @inproceedings{finn,
    author = {Mohammad Ghasemzadeh, Mohammad Samragh, Farinaz Koushanfar},
    title = {ReBNet: Residual Binarized Neural Network},
    booktitle = {Proceedings of the 26th IEEE International Symposium on Field-Programmable Custom Computing Machines},
    series = {FCCM '18},
    year = {2018}
    }

### 2. PYNQ-Classification

If you make use of this code, please acknowledge us by citing [our article](https://spiral.imperial.ac.uk/handle/10044/1/57937):

    @inproceedings{Wang_FCCM18,
    author={E. Wang and J. J. Davis and P. Y. K. Cheung},
    booktitle={IEEE Symposium on Field-programmable Custom Computing Machines (FCCM)},
    title={{A PYNQ-based Framework for Rapid CNN Prototyping}},
    year={2018}
    }

### 3. FINN

If you find BNN-PYNQ useful, please cite the <a href="https://arxiv.org/abs/1612.07119" target="_blank">FINN paper</a>:

    @inproceedings{finn,
    author = {Umuroglu, Yaman and Fraser, Nicholas J. and Gambardella, Giulio and Blott, Michaela and Leong, Philip and Jahre, Magnus and Vissers, Kees},
    title = {FINN: A Framework for Fast, Scalable Binarized Neural Network Inference},
    booktitle = {Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
    series = {FPGA '17},
    year = {2017},
    pages = {65--74},
    publisher = {ACM}
    }

