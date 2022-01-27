# MIL Model Agnostic Interpretability

This repo contains the code for "Model Agnostic Interpretability for Multiple Instance Learning".

### Overview

Executable scripts can be found in the `scripts` directory.
Source code can be found in the `src` directory.
One copy of each trained model can be found in `models`.
Outputs from experiments can be found in `out`.
Results can be found in `results`.

### Data

We use five custom data set implementations:
`mnist_bags.py`, `crc_dataset.py`, `sival_dataset`, `musk_dataset` `and tef_dataset`; 
all inherit from `mil_dataset.py`. 
Rather than returning a single instance, they return a bag of instances and a single label.

Sources:
* SIVAL: http://pages.cs.wisc.edu/~bsettles/data/
* MNIST: https://pytorch.org/vision/stable/datasets.html#mnist
* CRC: https://warwick.ac.uk/fac/cross_fac/tia/data/crchistolabelednucleihe/
* Musk: https://archive.ics.uci.edu/ml/datasets/Musk+%28Version+2%29
* Tiger, Elephant and Fox: http://www.cs.columbia.edu/~andrews/mil/datasets.html


### Models and training

The models are implemented in `src/model`.
We provide trained versions of these models in the models directory.
The training scripts are in `scripts/train`.
These can be used to train single or multiple models.
They were tuned using the scripts in `scripts/tune` .

### Interpretability

The interpretability functionality can be found in the `src/interpretability` directory.
The methods are implemented in `interpretability/instance_attribution`.

### Experiments

Our experiment scripts can be found in `scripts/experiments`.
These produce the sample size figures found in the paper.  
The output scripts can be found in `scripts/out`.
These produce the interpretability outputs found in the paper.  
The `milli_weights_plot` file produces the plots for the MILLI curve and integral.

### Running scripts

All paths are relative to the root of the repo, so scripts must be executed from this location.
Required libraries can be found in `requirements.txt`.
