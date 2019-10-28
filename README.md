# SLAC
Implementation of stochastic latent actor critic (SLAC, https://alexlee-gk.github.io/slac/) in pytorch. Note that CNNs are not implemented, but instead MLP layers.


## Requirements
- gym
- roboschool (optional)
- pytorch
- scipy (for saving results)

To run the program, just modify the hyperparameters and environment in learn_slac.py, and run 
```
python learn_slac.py
```
