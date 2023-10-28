# Graph Compilers


Deep learning models can be described as computational graphs.
Graph compilers take computational graphs as inputs, optimize the graph, and generate executable for target hardware.
[This blog](https://deci.ai/blog/graph-compilers/) discusses typical graph compiler optimizations.

### Operator Fusion
In deep learning, common operators are mapped to hardware-specific kernel implementations,
and hardware execute kernels sequentially according to the model definition.
Compilers leverage the knowledge of computational graph to **fuse operators** mainly to eliminate memory loading operations.
A common example is fusing batchnorm into the preceding convolution layer, mentioned in
[this blog](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/).
Batchnorm can be implemented as a matrix-vector operation, which can then be implemented as a $1 \times 1$ convolution.
Since convolution is commutative, we can rewrite the preceding convolution layer by
expanding the Batchnorm convolution and the original one.


### Operator Scheduling
Graph compilers schedule operators to maximize efficiency, espeically when the model is split between multiple devices.
Splitting a computational graph and parallelizing the execution enables efficient model size scaling,
a trend that is critical to the increasing capabilities of large language models.  
One notable example is [GSPMD](https://arxiv.org/abs/2105.04663), a compiler that parallelizes ML computation graphs.
ML framework JAX uses GSPMD for its multi-device computation, and the feature is known for being greatly user friendly.
[Here is a blog](https://irhum.github.io/blog/pjit/) that explains the parallelism in JAX.


## A friendly introduction to machine learning compilers and optimizers

[Blog post link](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)

This is a great introduction to machine learning compilers.
It motivates the need for compilers clearly and points out the fragmentation of hardware backends.
The blog post also talks a lot about operator fusion and ends with the author's hopefulness for WASM.


## Glow

[Paper Link](https://arxiv.org/abs/1805.00907)

Glow is a machine learning compiler by Meta.
Many of its features are seen in PyTorch 2, so I guess the PyTorch team drew inspirations from Glow.
Some features include:

### Multi-level IR
Programming language compilers struggle with optimizing loops,
so machine learning compilers need high-level constructs such as tensors and convolution operators to reason with.
To target multiple backends, Glow adds additional levels of intermediate representations (IR),
allowing multiple backends to share high-level optimization passes.

### Strongly-typed Graph
Glow's graph is strongly typed, which means every tensor shape is known at compile time.
This design decision does not seem to stand the test of time: Turns out real-world workloads have highly dynamic input and output shapes.
This is why PyTorch 2 is pushing to support dynamic shapes ([Documentation](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)).

### Node Lowering
When lowering graphs to hardware-specific operators, Glow does not map high-level operator nodes to hardware, e.g. fully-connected layer,
but further lowers the nodes to linear-algebra-level operators, e.g. a matrix multiplication and a broadcast addition.
This gradual lowering technique is also seen in PyTorch 2 ATen IR and Prim IR
(See my [blog post](https://github.com/kimbochen/md-blogs/tree/main/pytorch-systems-intro#the-pytorch-20-compiling-pipeline) for more).

### Quantization
Glow performs model weight quantization, which is something I find quite interesting.
Glow profiles the model inference by injecting profiling nodes to record the value range of activations.
The model is then recompiled with rescaling nodes and quantized operations with the profile information.
Glow also optimizes for quantization, including minimizing the number of precision conversions and folding conversions into operations.

### Operator Stacking
Operator stacking is essentially Glow's automatic kernel fusion.
To preserve CPU's memory bandwidth and capacity, Glow optimizes the graph by stacking operators that operate on different data (i.e. data parallel)
and perform them back-to-back on one piece of data.
Figure 8 of the paper shows an example that uses AVX instructions, so [here is a reference](https://hjlebbink.github.io/x86doc/) for x86 assembly.


## Operator Fusion in XLA

[Paper link](https://arxiv.org/abs/2301.13062)

This paper dives deeply into how XLA, the machine learning compiler behind TensorFlow and JAX, performs operator fusion.
I find the limitations that the author found quite interesting:

### Fusion opportunity relies heavily on the frontend code quality
This is quite surprising as I always thought frontend model code are relatively static and should not affect too much.
According to the author's experiments, removing a seemingly irrelevant concatenation operation resulted in 3.41$\times$ speedup.

### Conservative optimizations
The authors found XLA's rule-based optimizations are generally more conservative and skipped some optimization opportunities.
In addition, the author calls for auto-tuning loop unrolling, since the iterative nature of machine learning training brings plenty performance gains.
Finally, XLA does not seem to consider backpropagation-specific optimizations such as activation checkpointing, aka rematerialization,
when performing operator fusion. Activation checkpointing is implemented in AOT Autograd of PyTorch 2
([Explanation by Horrace He](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467).

### Custom CUDA kernel hinders optimization
Because XLA uses third-party library, e.g. cuDNN and cuBLAS, for specific ML operators without knowledge of the implementations,
XLA is unable to perform further fusion and is forced to map to the custom kernels, introducing additional layout conversion overhead.
The author's experiments supports this by showing that a CUDA-only implementation is 2.7$\times$ faster than XLA optimized one
because XLA launches 2 extra kernels.


> Hardware architecture
> This is not mentioned in the paper, but the efficacy of operator fusion is also dependent on the hardware architecture.
> Fusion only makes sense when the operands exceed the memory capacity.
> In other words, there is no point fusing kernels if everything, including intermediate values, can be accessed from high-speed memory.
> This can be a problem when new hardware increases memory capacity.


## Conclusion
As deep learning moves out of research setting into production enviornments, we need every possible optimization to run models efficiently.
There seems to be still a lot of optimization opportunities and definitely a long way from a unified compiler stack.

## Further Readings
- [Deep Learning Compiler Survey](https://arxiv.org/abs/2002.03794)
- [Intel OneDNN Graph Compiler](https://arxiv.org/abs/2301.01333)
- Groq, an ML hardware accelerator startup, has a compiler that does not generate or use kernels, which is quite unique (source to be added).
