# PointNet
A pytorch implementation of [arxiv:1612.00593](https://arxiv.org/abs/1612.00593)

Code are adapted from [nikitakaraevv/pointnet](https://github.com/nikitakaraevv/pointnet) and the official implementation [charlesq34/pointnet](https://github.com/charlesq34/pointnet) and tested with `python3.6`,`torch1.7.1+cu110`,`CUDA-11.0`

# Pointcloud Classification

Download the [*ModelNet10*](3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip) dataset and extract it to the root path.
## Install required libraries

    pip install trimesh h5py
## Preprocessing

    python preprocess.py

## Training

    python train.py