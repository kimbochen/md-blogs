---
title: "How I Structure My ML Projects - Pytorch Lightning"
date: 2021-09-14T15:57:42+08:00
draft: false
---

In this article, I share how I structure my projects with PyTorch Lightning, and I provide a 
template for those who are interested.

## Table of Contents
- [Introduction](#introduction)
- [Folder Structure](#folder-structure)
- [Training Script Overview](#training-script-overview)
- [How I deal with Hyperparameters](#how-i-deal-with-hyperparameters)
- [`LightningModule` Methods](#lightningmodule-methods)
  - [`step`](#step)
  - [`training_step` and `validation_step`](#training_step-and-validation_step)
  - [`configure_optimizers`](#configure_optimizers)
- [Argument Parser](#argument-parser)
  - [`Trainer` arguments in command line](#trainer-arguments-in-command-line)
  - [Set default value for `Trainer` arguments](#set-default-value-for-trainer-arguments)
  - [The `--gpus` flag](#the---gpus-flag)
  - [Additional logic](#additional-logic)
  - [Creating `Trainer` object with args](#creating-trainer-object-with-args)
- [Logger](#logger)
  - [Creating a logger](#creating-a-logger)
  - [Using the logger](#using-the-logger)
  - [Using TensorBoard](#using-tensorboard)
- [Model](#model)
- [Data](#data)
- [Conclusion](#conclusion)

## Introduction

In this article, I will explain how I structure my ML projects with PyTorch Lightning.  
Here's a full-fledged [example](https://github.com/kimbochen/panodpt), 
and you can refer to this article for explanation.  
I will assume that the reader is familiar with PyTorch and Python in general.  
That said, if you have questions, please do not hesitate to reach out to me at 
Twitter [@KimboChen](https://twitter.com/KimboChen). 

## Folder Structure

My project folder looks like this:
```
project
|_ data/
   |_ __init__.py
|  |_ dataset.py
|  |_ <data-related>.py
|
|_ model/
   |_ __init__.py
|  |_ backbone.py
|  |_ decoder.py
|  |_ <model-related>.py
|
|_ notebooks/
|  |_ experiment.ipynb
|  |_ <all-jupyter-notebooks>.ipynb
|
|_ trainer.py
```

`data` contains definitions of datasets and data transforms, 
`model` contains components of the model, and `notebooks` contains all Jupyter notebooks.
If there are any other modules that does not belong to either `data` or `model`, such as 
implementations of metrics, I would put them in a file `util.py`. 
`trainer.py` is where PyTorch Lightning comes into play. 
I will explain how each component is arranged in `trainer.py`.

## Training Script Overview

The training script contains a `LightningModule` and a `main` function.  
The `LightningModule` is essentially the whole system, containing the training logic.  
The `main` function is executes when the script `trainer.py` is run. 
It configures a `Trainer` object, creates a logger, and instantiates a model and dataloaders.  
Training starts when this is run: `trainer.fit(model, train_dl, val_dl)`.

## How I deal with Hyperparameters

Training an ML model usually has a lot of hyperparameters. 
As the main component of the system, the argument list of the `LightningModule` can easily be 
bloated.  
To tackle this problem, my general rule is: **Hardcode hyperparameters until they need to be 
tuned**. Specifically, I hardcode them as constants and put them below the imports. This has two benefits: 
- Be clear that what hyperparameters exist and can be tuned.
- Reduce arguments passed around, for hyperparameters can be accessed everywhere in the file.

When I have to tune a hyperparameter, I will add it to the argument parser and the logger.

## `LightningModule` Methods

### `step`

This is a function I use to factor out the shared parts in the training and the validation step. 
The input of the function would be the input of the model and the ground truth. 
It will return everything a loss function needs to compute the loss.

### `training_step` and `validation_step`

> function signature: `training_step(self, batch, idx)`

Argument `batch` is the output of the dataloader, which would be the input of step. 
Argument `idx` is the index of the batch, which is useful information for logging. 
In this function, we call `step`, call loss function, and log output, and return the loss.

### `configure_optimizers`

This function takes no arguments.
In this function, I would create the optimizer and the scheduler. 
For the optimizer, we can get the parameters of the model by calling `self.parameters()`. 
(`LightningModule` mostly behaves like `nn.Module`.) 
There are many ways of returning the objects instantiated. 
- If you only have 1 optimizer: simply `return optimizer`.
- If you have a scheduler and an optimizer: return them in lists, one for the optimizer, 
  one for the scheduler.
  ```python
  return [opt1, opt2]  # 2 optimizers
  return [opt], [sch]  # 1 optimizer, 1 scheduler
  return [opt1, opt2], [sch1, sch2]  # 2 optimizers, 2 schedulers
  ```
  For more complex scenarios to pass optimizers, consider reading [the docs](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=configure_optimizers#configure-optimizers).

## Argument Parser

I use the argument parser for two purposes:
1. Providing command-line arguments that I defined, such as GPU IDs.
1. Providing `Trainer` arguments that I may use occasionally.

### `Trainer` arguments in command line

`Trainer` actually comes with an extensive amount of useful option flags. 
Thus, to use them as command-line arguments, I would use this:
```python
import pytorch_lightning as pl
parser = pl.Trainer.add_argparse_args(parser)
```

### Set default value for `Trainer` arguments

There are `Trainer` arguments that are set once and for all, 
so it would be tedious to type them repeatedly in the command line. 
To solve this problem, I assign values to arguments **after** they are parsed.
```python
args = parser.parse_args()
args.deterministic = True
```

### The `--gpus` flag

`Trainer` provides a `---gpus` flag to specify the number of GPUs or the IDs of GPUs. 
Nevertheless, I struggle making the flag represent GPU ID 1 in command line.
My workaround is creating my own and assigning it to `args.gpus`.
```python
parser.add_argument('--gpu', '-g', nargs='+', type=int, default=[0, 1])
...
args.gpus = args.gpu
```

### Additional logic

Sometimes certain arguments are used together. 
Here are two scenarios that I use:
```python
# If more than 1 gpu specified, turn on distributed data parallel
args.accelerator = 'ddp' if len(args.gpus) > 1 else None

# Create a logger only if the run is not a test run
if not args.fast_dev_run:
    args.logger = pl.loggers.TensorBoardLogger(...)
```

### Creating `Trainer` object with args

To create a trainer,
```python
trainer = pl.Trainer.from_argparse_args(args)
```
`from_argparse_args` ignores unknown arguments, 
so beware of spelling mistakes in the arguments that are set above.

## Logger

### Creating a logger

> `pl.logger.TensorBoardLogger(save_dir, name, version)`

One experiment would be saved as the following structure:
```
save_dir/
|_ name/
   |_ version/
      |_ hparams.yaml
      |_ events.out.tfevents.*
      |_ checkpoints/
         |_ *.ckpt
```
- I mostly use `name` as the common setting of a series of compared experiment, 
  e.g. trained on 390 samples.
- I name `version` after what this experiment is different from others, 
  such as what hyperparameter is set, or model changes.

### Using the logger

#### Logging hyperparameters

The **only** way I know that works is the following:
```python
logger = pl.loggers.TensorBoardLogger(...)
logger.log_hyperparams(...)
```

Contrary to what the documentation says, 
`logger.log_hyperparams` does not work in the `LightningModule` in my experience.  
What's worse is that it fails silently, 
making it excrutiatingly difficult to know what went wrong. 
After plowing through the source code of PyTorch Lightning and reading related GitHub issues, 
I still fail to comprehend the mechanism. 
_If you know the reason, please consider telling me._ 
If you simply want to use what PyTorch Lightning offers, you can use what is mentioned above.

#### Logging in loss and metrics

I use `LightningModule`'s built-in API `self.log` instead of calling TensorBoard functions. 
It simplifies code and eliminates the need to calculate which step to log data.
```python
# In __init__
self.log = partial(self.log, on_step=False, on_epoch=True)
```

Here is an example of the visualization on TensorBoard.![](/tensorboard_vis.png)
I use the tags to visualize metrics of training and validation side by side. Concretely, 
```python
# In training_step
self.log('Delta1/Train', metric)
self.log('MAE/Train', loss)

# In validation_step
self.log('Delta1/Val', metric)
self.log('MAE/Val', loss)
```

### Using TensorBoard

For launching TensorBoard, I use this command:
```bash
tensorboard --logdir <experiment folder, i.e. name> --port <port number>
```

To share experiments with others, I upload it online:
```bash
tensorboard dev upload --logdir <experiment folder, i.e. name>
```

To delete experiments online:
```bash
tensorboard dev delete --experiment_id <experiment_id>
```

As of TensorBoard 2.4.1 with reduced feature set, data uploaded cannot be changed unless 
the experiment is deleted.

## Model

The definition of the model is put in the constructor of a `LightningModule`. 
Components of the model are imported from the package `model`.  
Like a `nn.Module`, we also define a `forward` function. 
I recommend doing so because we can then use `LightningModule` for inference. 

```python
class LitModel(pl.LightningModule):
    def __init__(self):
        self.backbone = Backbone(...)
        self.decoder = Decoder(..)
    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x
# Inference
model = LitModel.load_from_checkpoint_path(...)
model.eval()

with torch.inference_mode():  # Extreme version of `torch.no_grad`
    yb = model(xb)
```

## Data

Training and validation datasets/dataloaders usually have many shared arguments. 
To simplify code, I would use `functools.partial`, 
which creates functions with some arguments specified. 
Concretely, 
```python
from functools import partial
dl = partial(torch.nn.DataLoader, batch_size=9, num_workers=8, pin_memory=True)
train_dl, val_dl = dl(MyDataset('train')), dl(MyDataset('val'))
```

## Conclusion

In this article, I explain how I use PyTorch Lightning and my design choices. 
I hope it helps those who want to try out PyTorch Lightning, but are not sure where to start.  
Please feel free to share this or discuss with me at Twitter [@KimboChen](https://twitter.com/KimboChen). On the other hand, if you have any suggestions on my writing or more tips of using PyTorch Lightning, I would love to hear them!
