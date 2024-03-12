---
title: "How I Structure My ML Projects - PyTorch"
date: 2021-09-16T10:37:39+08:00
draft: false
---

This is the second part of a two-part series _How I Structure My ML Projects_.  
In the [first part](https://kimbochen.github.io/posts/structure-ml-projects-pytorch-lightning/), 
I shared how I structure my projects with the library PyTorch Lightning.  
In this article, I would like to share some tips on using PyTorch itself and some basic Jupyter Notebook usage.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Modules That are in `data`](#modules-that-are-in-data)
  - [Hard-code dataset path](#hard-code-dataset-path)
  - [Always test data-related code after developing it](#always-test-data-related-code-after-developing-it)
  - [Check your data types](#check-your-data-types)
- [Modules That are in `model`](#modules-that-are-in-model)
  - [Hardcode hyperparameters by default](#hardcode-hyperparameters-by-default)
  - [Use keyword arguments](#use-keyword-arguments)
  - [Use `nn.Sequential` to chain up modules](#use-nnsequential-to-chain-up-modules)
  - [Use `nn.ModuleList` when creating a list of `nn.Module`s](#use-nnmodulelist-when-creating-a-list-of-nnmodules)
  - [Beware of using torch.nn.functional](#beware-of-using-torchnnfunctional)
  - [Use `nn.Identity` to reduce forward logic](#use-nnidentity-to-reduce-forward-logic)
- [How I use Jupyter Notebooks](#how-i-use-jupyter-notebooks)
  - [Setup](#setup)
  - [Usage](#usage)

## Folder Structure

Just for a quick recap, here's the folder structure:

```
project/
|_ data/
|_ model/
|_ notebooks/
|_ trainer.py
```

## Modules That are in `data`

Modules in folder `data` are data-related. 
This includes the definition of the dataset, and implementations of data transforms. 
Here are some design rules and tips that I use:

### Hard-code dataset path

I write one `class` for every dataset, so there is no point making the dataset path an argument 
of the constructor. Writing specific code improves code clarity and reduces errors. 

### Always test data-related code after developing it

Making sure the right thing is put into the model is extremely important. 
Visualize and check the output of the dataset object, which is the data just before entering the 
model.

### Check your data types

The default data type for NumPy is `float64`, whereas most models use `float32` or lower. 
- When creating NumPy arrays, use `np.float32` instead of `np.asarray`. (Strangely, I cannot 
  find the documentation of `np.float32`, but it works flawlessly for me.)
- If you cannot create data as `float32` and happen to work with images, `torchvision` provides a 
  transformation that converts data type. Usage: 
  ```python
  import torchvision.transforms as T
  transforms = T.Compose([
      T.ConvertImageDtype(torch.float32),
      T.OtherTransformations,
      ...
  ])
  x = ...  # <data that comes with other data types>
  y = transforms(x)
  ```

## Modules That are in `model`


Definitions of model **components**, but not the model itself, are in the `model` folder. 
The model definition is taken care of by the `LightningModule` instead, more on that later.

Models usually have a lot of hyperparameters to tune, so it can quickly make the constructor 
argument list too long. Here are some tips to combat this:

### Hardcode hyperparameters by default

I usually hardcode hyperparameters by **making them a constant in the file**, and only put them 
in the argument list when I need to tune it. For instance,
```python
IN_CH, OUT_CH = 39, 85

class MyFullyConnected(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(IN_CH, OUT_C)
```

### Use keyword arguments

I use keyword arguments for ones that are actually for submodules. 
With keyword arguments, we do not need to change the arguments of the parent module if the 
submodules are replaced. Concretely,
```python
class ParentModule(nn.Module):
    def __init__(self, in_dim, **kwargs):
        self.linear = nn.Linear(in_dim, in_dim)
        self.child_mod = MyChildModule(**kwargs)
```

---

After specifying arguments, we usually construct a model using `torch.nn`. 
Here are some tips on doing so:

### Use `nn.Sequential` to chain up modules

When modules are only used together, I use `nn.Sequential` to chain them up for better 
readability. e.g.
```python
'''GOOD'''
# In `__init__`
self.conv = nn.Conv2d(...)
self.norm = nn.BatchNorm2d(...)
self.activation = nn.ReLU()
# In `forward`
x = self.activation(self.norm(self.conv(x)))

'''BETTER'''
# In `__init__`
self.layer = nn.Sequential(
    nn.Conv2d(...)
    nn.BatchNorm2d(...)
    nn.ReLU()
)
# In `forward`
x = self.layer(x)
```
However, overuse of `nn.Sequential` may backfire and hurt readability. 
So I would recommend using it sparingly.

### Use `nn.ModuleList` when creating a list of `nn.Module`s

`nn.ModuleList` can correctly encapsulate the parameters of modules for backprop. 
I once encountered an error that took me forever to debug, and replacing Python list with 
`nn.ModuleList` solved the problem.

### Beware of using torch.nn.functional

`nn` layers often have functional counterparts. 
However, `nn` layers usually take care of more details. 
For instance, `nn.Dropout` and `functional.dropout` are equivalent, 
but `nn.Dropout` automatically turns off during inference, while `functional.dropout` does not. 
So use functional forms cautiously. 

### Use `nn.Identity` to reduce forward logic

`nn.Identity` is a no-op that just passes input forward. 
Sometimes our `forward` function involves if else branches based on a predetermined value, 
we can do something like this:
```python
'''GOOD'''
# In `__init__`
self.conv = nn.Conv2d(...)
if norm:
    self.norm = nn.BatchNorm2d(...)
self.activation = nn.ReLU()
# In `forward`
x = self.conv(x)
if self.norm is not None:
    x = self.norm(x)
x = self.activation(x)

'''BETTER'''
# In `__init__`
self.layer = nn.Sequential(
    nn.Conv2d(...),
    nn.BatchNorm2d(...) if norm else nn.Identity(),
    nn.ReLU()
)
# In `forward`
x = self.layer(x)
```

## How I use Jupyter Notebooks

I use Jupyter notebooks for visualization and testing. 
Here is how I use them.

### Setup

To eliminate the problem of importing functions, I set `PYTHONPATH` as the folder. 
Specifically, I use the following command under folder `project`:
```
PYTHONPATH=`pwd` jupyter notebook --no-browser --port 3985
# `pwd` is command for current directory
```
I would then connect to the server by entering this command on my laptop:
```
ssh -fNL 3985:localhost:3985 <username>@<server_ip>
# -f: background execution
# -N: Just forward ports
# -L: port forwarding
```
_Full disclosure: I still struggle to understand how port forwarding in ssh works. 
If you happen to know, consider offering me some explanation or recommending me some resources._


### Usage

- Visualizing images: I transpose tensors to shape `C x H x W` (Channel first), 
  and use `torchvision.transforms.ToPILImage` to visualize the image.
- I also use Jupyter notebooks to check tensor shapes and equivalence with `torch.allclose`.
- I use `torchinfo.summary` to check the overall input and output shape of a model.
- I use pdb for debugging in general.
  - I simply insert a one-liner in the place I want to debug:
    ```
    import pdb; pdb.set_trace()
    ```
  - Honestly, pdb in command-line interface is better, since bash shortcuts are supported.
