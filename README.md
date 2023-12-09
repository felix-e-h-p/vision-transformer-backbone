# Vision Transformer (ViT) Training

## Overview

This repository contains a distributed training script for Vision Transformer (ViT) models using PyTorch. The script supports parallel training across multiple GPUs using PyTorch's Distributed Data Parallelism.

## Requirements

- Python 3.x
- PyTorch
- wandb (for logging and visualization)
- Other dependencies (check the script and install as needed)

## Usage

### Training

To train the ViT model, run the following command:

```bash
python train.py --world_size <num_gpus> --mode train
```
### Validation

```bash
python train.py --world_size <num_gpus> --mode val
```
