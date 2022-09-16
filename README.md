# A-VB Emotions

This repo contains code to reproduce our contribution for the ACII A-VB 2022 Competition and Workshop.

https://www.competitions.hume.ai/avb2022

We address all 4 tasks of the competition with a multi-task learning approach based on loss balanding and using SSL audio models as shared feature extractors.

## Installation

Runs on Python >=3.9.

We recommend using conda to install the dependencies. 

Required packages:

liac-arff
librosa >= 0.8.1
matplotlib
numpy
pip
pandas >= 1.4.0
pytorch-lightning
ray
scikit-learn
scipy
seaborn
tabulate
tensorboard
tensorboardx
torch
torch-tb-profiler
torchaudio
tqdm
transformers
pytorch-model-summary
pynvml

Also, you will need to install the End2You toolkit https://github.com/end2you/end2you. 


## Training

The main script is train.py, which instantiates a Trainer object and runs its train() and test() functions.

Construct a model from one of the architecture choices, pick a loss balancing strategy and train it on the A-VB dataset, then evaluate on the val and test sets. 

You can log the training progress to TB, and prediction csv files for each task will be generated at the end of the evaluations.

There are many options which can be customised via CLI. Alternatively, you can create an End2You Params object from a config file and pass it to the Trainer class.




