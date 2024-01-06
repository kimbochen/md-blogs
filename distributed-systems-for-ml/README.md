# Distributed Systems for Machine Learning

This is a live flow of my process of learning about distributed systems for machine learning.


## Table of Contents
- [My Impressions before Diving In](#my-impressions-before-diving-in)
- [ZeRO and DeepSpeed](#zero-and-deepspeed)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](#zero-memory-optimizations-toward-training-trillion-parameter-models)
- [Megatron-LM](#megatron-lm)
- [Weight Streaming](#weight-streaming)
- [PyTorch Fully Sharded Data Parallel](#pytorch-fully-sharded-data-parallel)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](#pytorch-fsdp-experiences-on-scaling-fully-sharded-data-parallel)
- [Automatic Mixed Precision Gradient Scaling](#automatic-mixed-precision-gradient-scaling)
- [Collective Operations](#collective-operations)
- [General and Scalable Parallelization for Neural Networks](#general-and-scalable-parallelization-for-neural-networks)
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](#gspmd-general-and-scalable-parallelization-for-ml-computation-graphs)
- [PyTorch/XLA SPMD: Scale Up Model Training and Serving with Automatic Parallelization](#pytorchxla-spmd-scale-up-model-training-and-serving-with-automatic-parallelization)
- [High-Performance Llama 2 Training and Inference with PyTorch/XLA on Cloud TPUs](#high-performance-llama-2-training-and-inference-with-pytorchxla-on-cloud-tpus)
- [Lightning Talk: Large-Scale Distributed Training with Dynamo and PyTorch/XLA SPMD](#lightning-talk-large-scale-distributed-training-with-dynamo-and-pytorchxla-spmd)
- [Pathways: Asynchronous Distributed Dataflow for ML](#pathways-asynchronous-distributed-dataflow-for-ml)
- [Tensor Parallelism with `jax.pjit`](#tensor-parallelism-with-jaxpjit)
- [How to scale AI training to up to tens of thousands of Cloud TPU chips with Multislice](#how-to-scale-ai-training-to-up-to-tens-of-thousands-of-cloud-tpu-chips-with-multislice)
- [Project Fiddle: Fast and Efficient Infrastructure for Distributed Deep Learning](#project-fiddle-fast-and-efficient-infrastructure-for-distributed-deep-learning)


## My Impressions before Diving In

Keywords:

- ZeRO
- Megatron-LM
- DeepSpeed
- Fully-sharded data parallel
- Sharded tensor
- GSPMD compiler XLA
- JAX distributed training mechanism
- Data parallel, model parallel, tensor parallel
- 3D parallelism
- GPT Neo distributed training
- Singularity
- Distributed training on low-bandwidth interconnect / commerical clusters
- TikTok off-loading training to inference cluster


## ZeRO and DeepSpeed

Blog post: [ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

- Published in February 2020, _over 100 billion parameters_ is what is needed to train GPT-3.
- ZeRO stands for Zero Redundancy Optimizer.
- The video around the middle of the blog post pretty much explains how ZeRO works.
- ZeRO-powered data parallelism: Allows per-device **memory usage** to **scale linearly** with the data parallelism **degree**.
  - Scaling memory usage: Increase? Decrease? **Answer: reduction (decrease).**
  - Data parallelism degree? **Number of GPU nodes processing different data batches.**
  - In other words, memory reduction is linear with the number of GPU nodes processing different data batches.
- The big idea of ZeRO:
  - Uses data parallel: Each node consumes a unique data batch
  - Each node holds a unique shard (partial copy) of the model states: Model parameters, optimizer states, gradients, and activations
  - Nodes exchange data when needed, e.g. all gathering model parameters for forward pass.
- DeepSpeed is a distributed training library that supports ZeRO.


## ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

Paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

- DP: Data parallelism, MP: Model parallelism
- DP vs. MP
  | | Data Parallelism | Model Parallelism |
  | :- | :- | :- |
  | Computation Granularity | Unchanged | Reduced |
  | Communication Overhead | Increased | Increased More |
  | Model States Distribution | Redundant | Partitioned |
- Insight: **Not everything in the model states is required all the time** $\implies$ [Weight streaming](#weight-streaming)?
- Standard memory usage - Mixed-precision training with Adam optimizer
  - FP16 model parameters + FP16 gradients + Optimizer states (FP32 parameters + FP32 momentum + FP32 variance)
  - Let $\Psi$ denote the number of model parameters $\implies$ memory usage $= 2 \Psi + 2 \Psi + (4 \Psi + 4 \Psi + 4 \Psi) = 16 \Psi$ bytes
- Memory savings: Partition to $N_d$ GPU nodes $\implies$ divide by $N_d$


## Megatron-LM

Paper: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- Main novelty: A model-parallel transformer
- Split the weight matrix along the column axis, allowing the GeLU non-linearity to be performed independently $\implies$ Tensor parallelism?
- Forward pass function is identity, then the backward pass function is an allreduce; Forward pass is allreduce, then backward pass is identity


## Weight Streaming

Blog Post: [Linear Scaling Made Possible with Weight Streaming](https://www.cerebras.net/blog/linear-scaling-made-possible-with-weight-streaming)

- The gif explains weight streaming pretty clearly:
  ![](https://www.cerebras.net/wp-content/uploads/2022/09/Weight-streaming-in-action-3.gif)
- The general idea is that we use specialized components, one for compute and one for weight updates.
  We stream weights to compute units from the weight update units.
- [Weight streaming whitepaper](https://8968533.fs1.hubspotusercontent-na1.net/hubfs/8968533/Virtual%20Booth%20Docs/CS%20Weight%20Streaming%20White%20Paper.pdf)


## PyTorch Fully Sharded Data Parallel

Blog post: [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)

- FSDP is basically the same as ZeRO stage 3
- This image is really helpful for understanding reduce-scatter:
  ![](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png?resize=768,422)
- This is standard data parallel training:
  ![](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?resize=768,867)
- This is fully sharded data parallel training:
  ![](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png?resize=768,867)
- FSDP performs allgather on model weights to do forward and backward pass
- FSDP performs reduce-scatter on gradients to syncchronize gradients:
  - Reduce across data batch axis: accumulate gradients in each node
  - Scatter across model layer axis: partition gradients to nodes that hold the model weights
- Another FSDP workflow image from blog post [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/):
  ![](https://pytorch.org/assets/images/fsdp_workflow.png)


## PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

Paper: [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)

- The paper shares a lot more details on the user experience and engineering designs of PyTorch FSDP
- Model partitioning: Splitting model into layers. Model sharding: Splitting layers into parameters
- Sharding algorithm: flatten-concat-chunk
  - Flattens layer parameters into a 1D-tensor wrapped in a `FlatParameter` module
  - Pads to maximize NCCL collective efficiency
- `FlatParameter` modules own the tensor storage; original layers are views into `FlatParameter`
- Communication optimization: overlapping communication and computation, backward and forward pre-fetching, accumulation
- Because of PyTorch only supports eager execution then, for forward pass FSDP cannot know what to allgather before issuing computation
- Backward prefetching: FSDP performs allgather for layer $i+1$ first and overlaps backward computation with reduce-scatter layer $i$ gradients
  - FSDP records forward pass module execution order as reference because of eager execution
- Forward prefetching: Assuming the model has a static computational graph, FSDP can use forward module execution order of previous iteration
  to overlap allgather with computation
- In hindsight, it is clear that compiling the program can help FSDP a lot with overlapping communication and computation
  - I wonder if FSDP in PyTorch 2 is able to leverage this now
  - This shows the advantages of frameworks that compiles model graph, e.g. JAX and TensorFlow
- Gradient accumulation: Lowering the number of reduce-scatter operations by doing gradient accumulation on all nodes
- FSDP sharding gradients across ranks causes **local gradient scaler breaking mathematical equivalence**

## Automatic Mixed Precision Gradient Scaling

- [PyTorch AMP gradient scaling documentation](https://pytorch.org/docs/stable/amp.html#gradient-scaling)
- [PyTorch AMP gradient scaler code](https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/grad_scaler.py)

- Gradient scaling scales the loss for the backward pass, so intermediate gradients are not lost due to underflow
- Gradients are **unscaled** when updating the parameters so it does not interfere with the learning rate
- The scale factor is dynamically estimated for each iteration and in FSDP possibly unique to each data batch, hence _breaking mathematical equivalence_


## Collective Operations

[NCCL 2.19 Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

- Allreduce := reduce + broadcast
- Allreduce := reduce-scatter + allgather


## General and Scalable Parallelization for Neural Networks

Blog post: [General and Scalable Parallelization for Neural Networks](https://blog.research.google/2021/12/general-and-scalable-parallelization.html)

- GSPMD is an automatic parallelization system based on the ML compiler XLA
- The main feature of GSPMD is that it separates model programming from parallelization
- Google touts GSPMD supproting data parallelism, pipeline parallelism, and operator-level parallelism, e.g. Mesh-TensorFlow
  - Operator-level parallelism sounds like tensor parallelism to me
- The parallelized matrix multiplication example is a bit weird to me but I believe it is largely the same as FSDP forward pass
- Nested parallelism example: GSPMD is capable of fine-grained parallelism strategy for different model components


## GSPMD: General and Scalable Parallelization for ML Computation Graphs

Paper: [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)

- Concept of sharding:
  - Replicated: All device have the same full data
  - Tiled: Each device has a tile (partition) of the data
  - Device mesh tensor: A tensor that maps data index to device ID (Reminds me of Triton GPU IR Layout Encoding)
  - Partially tiled: Tiled with device groups replicated
- $L$-staged pipeline parallelism converted to tensor sharding:
  - $L$ state tensors expressed as one $L \times \dots$ tensor
  - GSPMD vectorizes $L$ computation stages, executes them in parallel, and right shifts the tensor after every iteration
  - The tensor is then sharded and XLA will add collective communications as needed
  - Vectorization assumes identical computation stages
- I am having trouble understanding the notations, for example the ones in figure 3
- The rest of the papers seems to focus on how the compiler figures out how to shard. No mentioning of how devices communicate shards
- The compiler part is worth reading but maybe for another day
- GSPMD seems to have a more general programming model compared to FSDP and ZeRO: It deals with sharding tensors rather than model states
- GSPMD abstracts away how devices communicate, but lets users decide how to shard model states and data

> Further reading: [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) compiler sections


## PyTorch/XLA SPMD: Scale Up Model Training and Serving with Automatic Parallelization

Blog post: [PyTorch/XLA SPMD: Scale Up Model Training and Serving with Automatic Parallelization](https://pytorch.org/blog/pytorch-xla-spmd/)

- The graph makes more sense now:
  ![](https://pytorch.org/assets/images/pytorch-xla-spmd/fig1.png)
- Partition spec specifies which device mesh dimension should a dimension of the target tensor be partitioned like
- SPMD claims to outperform FSDP in terms of efficiency, measured in model FLOPs utilization
  ![](https://pytorch.org/assets/images/pytorch-xla-spmd/fig2.png)


## High-Performance Llama 2 Training and Inference with PyTorch/XLA on Cloud TPUs

Blog post: [High-Performance Llama 2 Training and Inference with PyTorch/XLA on Cloud TPUs](https://pytorch.org/blog/high-performance-llama-2/)

- [2D Sharding with SPMD](https://pytorch.org/blog/high-performance-llama-2/#2d-sharding-with-spmd) seems to be the most interesting
- Parameter sharding shows an example of how a user specifies partitions of the Llama 2 model
- Model-data axis rotation methodology is similar to that of Megatron-LM?
  - Megatron-LM splits self-attention operation along the head dimension so every device can perform self-attention independently
  - Here we shard the attention weights along the head dimension, and mapping it to the device mesh _model_ dimension means
    it will only shard for pipeline parallelism
- Oh my goodness optimizer states and gradients are sharded automatically!?
- MultiSlice seems like a relevant technology
- Inference optimizations also look really interesting
  but is less relevant. Maybe another day.

> Further reading: [Inference optimizations](https://pytorch.org/blog/high-performance-llama-2/#inference-optimizations)


## Lightning Talk: Large-Scale Distributed Training with Dynamo and PyTorch/XLA SPMD

YouTube video: [Lightning Talk: Large-Scale Distributed Training with Dynamo and PyTorch/XLA SPMD - Yeounoh Chung & Jiewen Tan, Google](https://www.youtube.com/watch?v=tWH2MAHzVVc)

- An example of the sharding notation! ([Image source with timestamp](https://www.youtube.com/watch?v=tWH2MAHzVVc&t=635s))
  ![](assets/2d-sharding-notation.png)
  - Subscript indicates the device mesh axis the tensor is mapped to.
- How XLA inserts collective communication operations ([Image source with timestamp](https://www.youtube.com/watch?v=tWH2MAHzVVc&t=721s))
  ![](assets/2d-sharding-partition.png)
  - Capital letters are tensor dimensions, lowercase letters are sharded tensor dimensions


## Pathways: Asynchronous Distributed Dataflow for ML

Paper: [Pathways: Asynchronous Distributed Dataflow for ML](https://arxiv.org/abs/2203.12533)

- _The rapid recent progress of ML has been characterized by the co-evolution of ML models, accelerator hardware,
  and software systems that tie the two together_
- State-of-the-art LM workloads are **single program multiple data**, but the current systems don't support well sparse models like Mixture-of-Experts
- Multi-controller architecture: PyTorch and JAX
  - The same client executable is run directly on all the hosts in the system
  - Takes exculsive ownership of host resources throught the program execution
  - All inter-host communication happen through collectives that use interconnects,
    e.g. [NVLink](https://ieeexplore.ieee.org/document/7924274), [ICI](https://dl.acm.org/doi/pdf/10.1145/3360307)
- Single-controller architecture: TensorFlow v1
  - A coordinator runtime partitions the computation graph into subgraphs and delegates them to workers
- The content is interesting but seems to be too _systems_ for me to understand (at least for now)
  - Keywords: Single-controller, multi-controller, gang-scheduled dynamic dispatch, parallel asynchronous dispatch,
    resource manager, client, coordination


## Tensor Parallelism with `jax.pjit`

Blog post: [Tensor Parallelism with `jax.pjit`](https://irhum.github.io/blog/pjit/)

- A really nice blog about how tensor parallelsim works
- Things start getting interesting from [Sharding: A matrix Multiply](https://irhum.github.io/blog/pjit/#sharding-a-matrix-multiply)
- **Why would we shard the matrix only to gather it in full on every single device? Because we can eliminate duplicate computation when fully sharded.**
- [GSPMD's sharding spec](https://irhum.github.io/blog/pjit/#gspmds-sharding-spec) section is an excellent walkthrough of an example of
  how GSPMD shards a tensor computation. The illustration is really helpful.

Further reading: [JAX `pjit` programming model](https://irhum.github.io/blog/pjit/#the-pjit-programming-model)


## How to scale AI training to up to tens of thousands of Cloud TPU chips with Multislice

Blog post: [How to scale AI training to up to tens of thousands of Cloud TPU chips with Multislice](https://cloud.google.com/blog/products/compute/using-cloud-tpu-multislice-to-scale-ai-workloads)

- Modeling system peak FLOPs with global batch size, DCN bandwidth, and chips per ICI domain
  - DCN: Data center network
  - ICI: Inter-chip interconnect
- For a dense LLVM using data parallelism or fully sharded data parallelism, DCN arithmetic intensity $\approx$ minimum batch size per ICI domain (Why?)
- XLA compiler understands hybrid DCN / ICI network topology and automatically inserts appropriate hierarchical collectives. Allreduce example:
  ![](https://storage.googleapis.com/gweb-cloudblog-publish/images/12_Tyktblk.max-2000x2000.jpg)


## Project Fiddle: Fast and Efficient Infrastructure for Distributed Deep Learning

Blog: [Project Fiddle: Fast and Efficient Infrastructure for Distributed Deep Learning](https://www.microsoft.com/en-us/research/project/fiddle/overview/)

- The goal is to build efficient systems infrastructure for very fast distributed DNN training
- There is a big publication list

Further reading: [Project Fiddle: Fast and Efficient Infrastructure for Distributed Deep Learning](https://www.microsoft.com/en-us/research/project/fiddle/publications/)
