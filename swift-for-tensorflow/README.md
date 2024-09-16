# Swift for TensorFlow

In Lex's podcast with Chris Lattner ([link](https://youtu.be/pdJQ8iVTwj8)),
Chris mentioned that his work on Swift for TensorFlow has influenced the design of PyTorch 2.0.
Because of this, I re-read the paper went down the rabbit hole of ML framework design and really enjoyed learning about it.
The [paper](https://arxiv.org/pdf/2102.13243.pdf) is full of gems: there are references to so many great papers.
In this post, I compiled all the resources I read and some brief explanations.

- [Automatic Differentiation](#automatic-differentiation)
- [Code Transformation](#code-transformation)
- [LazyTensor](#lazytensor)
- [Further Readings](#further-readings)


## Automatic Differentiation

The nomenclature of the autodiff mechanism in S4TF is similar to that of JAX, which is inspired by differential geometry.

Some resources of differential geometry:
- [Differential Geometry in Under 15 Minutes](https://youtu.be/oH0XZfnAbxQ)
- [MIT OCW Differential Geometry](https://ocw.mit.edu/courses/18-950-differential-geometry-fall-2008/)

Honestly, I haven't gotten gained the intuition behind the names, but here are 2 terms and their meanings in the context of machine learning.
- Primal / Tangent: Input / Output of the gradient function (["Hipster autodiff terminology" - Horace He](https://youtu.be/KpH0qPMGGTI?t=320))
- Vector-Jacobian Product: Reverse mode autodiff / Jacobian-Vector Product: Forward mode autodiff

The mechanism itself is taken from Autograd
([paper](https://dash.harvard.edu/bitstream/handle/1/33493599/MACLAURIN-DISSERTATION-2016.pdf?sequence=4&isAllowed=y),
[slides](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)), the precursor of JAX.


## Code Transformation

S4TF transforms code into differentiation code by recording the operations throughout the function, AKA tracing.
[Here's](https://www.tensorflow.org/guide/function#rules_of_tracing) a nice explanation of tracing,
and [here's](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf) a paper about how JAX does it.
S4TF performs the autodiff code transformation at ahead-of-time-compile time, which is also what PyTorch 2.0 `torch.compile` does.
On a side note, what Chris mentioned in the podcast, Graph Program Extraction,
is about how S4TF traces the code and compiles it into a TensorFlow graph.
The meeting title is [Graph Program Extraction and Device Partitioning in Swift for TensorFlow](https://youtu.be/HSneJdPkaKk).
According to Chris, this technique is used in PyTorch 2.0, so I guess TorchDynamo applies this technique to perform bytecode transformation.


## LazyTensor

LazyTensor is a technique that attempts to combine the flexibility of eager mode execution
and the performance optimizations of domain-specific compilers.  
The idea is that LazyTensor defers tensor computations until the program _observes_ the result tensor's content like eager mode execution does,
but **traces** the computation (Inspired by [JAX](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf))
for the compilers to optimize segments of code.
This feature is also implemented in PyTorch, and the researchers published a [paper](https://arxiv.org/abs/2102.13267)
and a [blog](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)) on this.


## Further Readings

A natural next step is to look into what compiler features PyTorch 2.0 has.
PyTorch 2.0 has 3 new components: TorchDynamo, AOTAutograd, and TorchInductor.
I wrote briefly about it in my
[PyTorch Systems Intro](/pytorch-systems-intro/README.md#The-PyTorch-20-compiling-pipeline) post,
but the 3 features are definitely worth a deep dive in the future.
