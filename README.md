# DPFN
[DPFN: Dual-Path Fusion Network for Lightweight Single Image Super-Resolution)

Jinliang Ou, Haoyu Zhang, Junheng Ge, Yuying He, Hanyu Gao, and Zhengnan Yin

## Environment

- [BasicSR = 1.4.2](https://github.com/XPixelGroup/BasicSR)

### Installation

```
pip install -r requirements.txt
python setup.py develop
```

## How To Test

- Refer to `./options/test/DPFN
- The pretrained models are available in `./experiments/pretrained_models`.
- Then run the follwing codes (taking `net_g_DPFN_x4.pth` as an example):

```
python basicsr/test.py -opt options/test/DPFN/test_dpfn_x4.yml
```

The testing results will be saved in the `./results` folder.

- Refer to `./inference` for **inference** without the ground truth image.
- Refer to `./basicsr/calculate_params_flops.py` for calculating the **parameters and flops.**

## How To Train

- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).
- The training command is like:

```
python basicsr/train.py -opt options/train/DPFN/train_dpfn_x4.yml
python basicsr/train.py -opt options/train/DPFN/ft_dpfn_x4.yml
```

More training commands can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md).

The training logs and weights will be saved in the `./experiments` folder.


## Contact

If you have any question, please email: oujinliang01@swfu.edu.cn.
