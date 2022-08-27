# Generation of the Hermite-Gaussian modes
This repo are contains of code for classification/regression/generation of Hermite-Gaussian modes. This repo were created during graduating from university, there my goal were to research modes - so there is many other stuff written on TensorFlow 2.4.

There are many generation (Generative Adversarial Networks) configuration and code for it. Vanila-GAN, GAN with project-discriminator and other GANs you can find in `different_gans` and `gan_tests` folders.

In `utils` - you can find some help-code for this repo. There are code - `maki_kfold.py` with `MakiKFoldBalance` which is same as SKLearn kfold class, but this implementation works for multiclass and provide balances batches of data (if dataset itself is balanced of course).

The dataset you can find in `double_modes.zip` folder. The dataset - its interpolation between two modes. Name of the folder inside dataset - its interpolation coefficient.

