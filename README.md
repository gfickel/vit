# Visual Transformers
Some small tests with [Vision Transformer](https://arxiv.org/abs/2010.11929). The idea here is to play around with several of its components and see the impact on some small datasets such as mnist and cifar100.

## Configurations
Here is a list of things of the most important things you can play around with:
- Image size: both mnist and cifar100 are quite small and this architecture benefits from an upsize.
- Patch size
- Switch on/off the positional encoding learning
- Projection dimensionality
- Some inner transformer configs

## Todo
- Implement [DeiT](https://arxiv.org/abs/2012.12877)
- Implement Multihead Attention by hand instead of using the one from TF

## Shoutouts
This code was heavily inspired from https://keras.io/examples/vision/image_classification_with_vision_transformer/
