# ProteinGLUE accompanying code

Primarily, this repository contains Python scripts and notebooks to:
- Pre-train the BERT model using the Pfam data set
- Fine-tune pre-trained BERT models on the various downstream tasks
- Evaluate performance metrics for the various downstream tasks

Additionally, some of the code can be used standalone to:
- Load a TFRecord file into a custom model
- Generate TFRecord files from other input formats for the pre-training and downstream tasks
- Run training on a SLURM job system cluster

## Requirements

- Python 3.7 or newer
- Tensorflow 2.7.0 or newer

## Installation

Clone this repository using Git, change the directory to the cloned repository root, then install the requirements (into a virtualenv) with the following command.

```shell
$ pip install -r requirements.txt
```

## Pre-training a model

Pre-training a version of our BERT-like model is done using the `pretrain.py` script. An example invocation is given below.
```shell
$ python pretrain.py bert-sequences-k1.tfrecord bert_medium --target gpu_mixed --num-epochs 25 --num-steps 10000000 --batch-size 256 --big-batch-size 16 --num-heads 8 --num-layers 8 --d-model 512 --predict-nop-rate 0.1 --predict-random-rate 0.1 --learning-rate 0.00025 --big-input-file bert-sequences-big-k1.tfrecord
```

For a full list of options which can be passed to the pre-training script, please provide the `-h` argument. The given options in the example invocation are described below.
- `bert-sequences-k1.tfrecord` - Load normal-sized input sequences from this path. Glob patterns such as `*` are supported.
- `bert_medium` - Checkpoint directory where the model is stored. This can be used later as an input to restart the pre-training, or to load a model into the fine-tuning script.
- `--target gpu_mixed` - Pre-train on the GPU, with the mixed float support enabled. See the source code for currently supported targets.
- `--num-epochs 25` - Number of epochs to train for. One epoch means all the input sequences were presented to the model for training once.
- `--num-steps 1000000` - Train for this number of steps. Further used to linearly decay the learning rate. The current training step is saved in the model checkpoint file, so restarting training will _not_ reset it.
- `--batch-size 256` - Number of normal-sized sequences to feed to the model per training step. This parameter is hardware dependent and will require some tuning to optimize.
- `--big-batch-size 16` - Number of big sequences to feed to the model during a big training step. This parameter is hardware dependent and will require some tuning to optimize.
- `--num-heads 8` - The number of attention heads to use.
- `--num-layers 8` - The number of BERT layers to use.
- `--d-model 512` - The model depth per position. This number must be divisible by `--num-heads`.
- `--predict-nop-rate 0.1` - For the masked symbol prediction objective, the fraction of symbols to pass through unmodified.
- `--predict-random-rate 0.1` - For the masked symbol prediction objective, the fraction of symbols to randomize.
- `--learning-rate 0.00025` - The LAMB optimizer learning rate. This also generally requires some tuning. A value that is set too low will hamper training performance, a value that is set too high will cause the training to become unstable.

## Fine-tuning and evaluation

Fine-tuning models for the downstream tasks is done with the `train_downstream.py` script. An example invocation is given below:

```shell
$ python train_downstream.py --task ss3 --tensorboard-dir bert_medium_ss3_metrics --validation-file val-ss3.tfrecord --pretrain-checkpoint-dir bert_medium train-ss3.tfrecord bert_medium_ss3
```

A quick explanation of the options used in the sample invocation are given below; again the `-h` option can be provided to the script to see a full list of supported options and arguments.
- `--task ss3` - The downstream task to fine-tune a model for.
- `--tensorboard-dir` - Path to a directory to write training / validation metrics to (in tensorboard format).
- `--validation-file` - Path to a file containing the validation set.
- `--pretrain-checkpoint-dir` - Path to a directory containing the checkpoint for a pre-trained model. If this is given, care must be taken that the architecture of the model being fine-tuned is equal to that of the pre-trained model.
- `train-ss3.tfrecord` - Path to a file containing the training set for the downstream task which has been selected.
- `bert_medium_ss3` - Path to a checkpoint directory to write the fine-tuned model to.


