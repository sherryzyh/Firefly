
# Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks

This repository is NOT the official implementation of Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks.
[[paper link]](https://arxiv.org/pdf/2102.08574.pdf)
[[official codes]](https://github.com/klightz/Firefly)

## Requirements

To run the code, please download the pytorch >= 1.0 with torchvision


## Training

To train the model(s) in the paper, run this command:

```train
python main.py --method fireflyn --model vgg19
```

You can also try different growing method [exact/fast/fireflyn/random] which represent original splitting, fast splitting, firefly splitting, NASH described in the paper.

## Improvements in this repo

- Any cuda device is allowed when assigned.
- A summary log file after training.
- Allow binary convolution in training.