# smolgrad
learning about AGs. This autograd engine was inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd). I wrote the code and created a couple of benchmarks to see how it would end up. 

## Benchmarks

### XOR Problem

The MLP built on top of the autograd engine handles `xor` effortlessly. In 200 steps, it achieves a 100% accuracy.

### Two Moons problem

The MLP built on top of the autograd engine struggles (computationally) with the amount of data from our _two moons_ dataset. The implementation was for a learning exercise so no optimizations were even explored. I just wanted to be able to mentally map how neural networks work.

It doesn't achieve a 100% accuracy with the standard 16, 16, 1 layer structure. I managed to get 100% accuracy (loss ~0.02) with a layout of 8, 16, 16, 1.

## Misc

There's also a helper tool written by AI to visualize the nodes in the network at each layer.
