# Triton GPU IR Analysis


## Table of Contents

- [Triton Compiler Code Structure](#triton-compiler-code-structure)
  - [Triton GPU IR](#triton-gpu-ir)
  - [Other Components](#other-components)
- [Triton GPU IR Attribute Definitions](#triton-gpu-ir-attribute-definitions)
  - [Distributed Layout Encoding](#distributed-layout-encoding)
  - [Blocked Layout Encoding](#blocked-layout-encoding)
  - [Other Layout Encodings](#other-layout-encodings)
- [Axis Information](#axis-information)
  - [Contiguity](#contiguity)
  - [Divisibility](#divisibility)
  - [Constancy](#constancy)
  - [Constraints](#constraints)
- [Optimize Thread Locality Pass](#optimize-thread-locality-pass)
  - [Get Thread Locality Optimized Encoding](#get-thread-locality-optimized-encoding)
  - [Get Thread Locality Optimized Shape](#get-thread-locality-optimized-shape)
- [Coalesce Pass](#coalesce-pass)
  - [AxisInfo](#axisinfo)
  - [Number of Threads Per Element](#number-of-threads-per-element)
- [Lowering Python Front-end code to GPU IR](#lowering-python-front-end-code-to-gpu-ir)


## Triton Compiler Code Structure

The code structure for the Triton compiler is as follows:

```bash
/include/triton/  # *.h, *.td
    Analysis/
    Conversion/
    Dialect/
/lib/triton/      # *.cpp
    Analysis/
    Conversion/
    Dialect/
```

The header files and the TableGen files are mostly under `/include/triton`,
while `/lib/triton` includes the implementations.
TableGen files generally include helpful concept explanations.

### Triton GPU IR

In this analysis, we will focus on the Triton GPU IR.
The Triton GPU IR is under the 2 `Dialect` folders:

```bash
/include/triton/Dialect/TritonGPU/
    IR/
        Dialect.h
        TritonGPUAttrDefs.td
        TritonGPUOps.td
    Transforms/
        Passes.td
/lib/triton/Dialect/TritonGPU/
    Transforms/
        Prefetch.cpp
        Coalesce.cpp
        OptimizeThreadLocality.cpp
        AccelerateMatmul.cpp
        OptimizeDotOperands.cpp
        OptimizeEpilogue.cpp
        ReorderInstructions.cpp
        DecomposeConversions.cpp
        RemoveLayoutConversions.cpp
```

`/include/triton/Dialect/TritonGPU/IR/` includes the Triton GPU IR's dialect, attributes, and operations.
Optimization pass names are defined in `Trasnforms/Passes.td`.  
`/lib/triton/Dialect/TritonGPU/Transforms` contains the all Triton GPU IR optimization passes.
They can be roughly grouped into 3 types: CUDA-related, tensor-core-related, and compiler-related.

- CUDA: These passes optimize for the CUDA programming model.
  - `Prefetch`: Prefetches memory for consecutive dot product operations, reducing memory access time.
  - `Coalesce`
  - `OptimizeThreadLocality`
- Tensor core: Theses passes optimize for the MMA architecture.
  - `AccelerateMatmul`
  - `OptimizeDotOperands`
  - `OptimizeEpilogue`
- Compiler: These passes optimizes for the instruction efficiency.
  - `ReorderInstructions`
  - `DecomposeConversions`
  - `RemoveLayoutConversions`

In future sections, I will be detailing the [`Coalesce`](#coalesce-pass) and the [`OptimizeThreadLocality`](#optimize-thread-locality-pass) pass.

### Other Components

These are other stuff in the compiler that are worth mentioning:

```bash
/include/triton/
    Analysis/
        AxisInfo.h
        Allocation.h
        Membar.h
    Conversion/
        TritonToTritonGPU/
            Passes.td
        TritonGPUToLLVM/
            Passes.td
    Dialect/
        Triton/
            IR/
                TritonOps.td
                TritonTypes.td
            Transforms/
                Passes.td
/lib/triton/
    Analysis/
        AxisInfo.cpp
        Allocation.cpp
        Membar.cpp
    Conversion/
        TritonToTritonGPU/
            TritonToTritonGPUPass.cpp
    Dialect/
        Triton/
            Transforms/
                Combine.cpp
                ReorderBroadcast.cpp
                RewriteTensorPointer.cpp
```

- `Analysis/`:
  - `AxisInfo`: The definition of a tensor's axis information.
  - `Allocation`: The shared memory allocation strategy.
  - `Membar`: The placement of shared memory barrier, i.e. `__syncthreads()`.
- `Conversion`: How dialects are lowered from one to another.
- `Dialect/Triton`: The Triton IR, the IR which Triton GPU IR lowers from.


## Triton GPU IR Attribute Definitions

The Triton GPU IR attributes are defined in `/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`.  
First, 2 CUDA term translations:

| Term | CUDA Term |
| :- | :- |
| CGA (Cooperative Grid Array) | Grid |
| CTA (Cooperative Thread Array) | Thread Block |


### Distributed Layout Encoding

Triton GPU IR attributes are all defining **layouts**, aka layout encodings.  
A layout is a function that maps a tensor index to the set of CUDA threads that can access the tensor element.  
Formally speaking, layout $\mathcal{L}: \mathbb{Z}^d \rightarrow \{ t | t \in \mathbb{Z}\}$

The base class for a layout is `DistributedLayoutEncoding`, which is defined with a tensor $T$.  
Let the data tensor be $A$ and its index be $i$, $rank(A) = rank(T) = D, i \in \mathbb{Z}^D$

Layout `L_T[i] = { T[idx(k)] }` $\forall k$, where
```python
idx(k) = [ (i_d + k * dim(A, d)) % dim(T, d) for d in range(D)]
```

For example,
```python
A.shape = [2, 8]
T = [
    [0,  1,  2,  3 ],
    [4,  5,  6,  7 ],
    [8,  9,  10, 11],
    [12, 13, 14, 15]
]
T.shape = [4, 4]
i = [1, 3]

idx(0) = [(1 + 0 * 2) % 4, (3 + 0 * 8) % 4] = [1, 3]
idx(1) = [(1 + 1 * 2) % 4, (3 + 1 * 8) % 4] = [3, 3]
L[i] = { L[idx(0)], L[idx(1)] } = { L[1, 3], L[3, 3] } = {7, 15}
```


### Blocked Layout Encoding

A layout can also be defined with the 4 hierarchical levels in the CUDA programming model:

- CTAs per CGA
- Warps per CTA
- Threads per warp
- Values per thread (Size per thread)

In each level, blocked layout encoding specifies a layout with `shape` and `order`,
where `order` means the axis iteration order, starting from the fastest changing axis.
For example,
```python
shape = [2, 3]; order = [0, 1]
layout = [
    [0, 2, 4],
    [1, 3, 5]
]
```

Here's an example of a blocked layout encoding of a $32 \times 32$ tensor:

| Hierarchy | Shape | Order |
| :- | :-: | :-: |
| `SizePerThread` | `[2, 2]` | `[1, 0]` |
| `ThreadsPerWarp` | `[8, 4]` | `[1, 0]` |
| `WarpsPerCTA` | `[1, 2]` | `[1, 0]` |
| `CTAsPerCGA` | `[2, 2]` | `[0, 1]` |

The layout function would be:
```python
CTA [0,0]                                              CTA [1,0]
[ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]
...                               ...          ...
[ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]

CTA [0,1]                                              CTA [1,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 ... 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 ... 39 39 ]
...                               ...          ...
[ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 ... 63 63 ]
```
(Example taken from
[the source code](https://github.com/openai/triton/blob/addd94e4a8d4dc0beefd5df6d97d57d18065436a/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L444).)


### Other Layout Encodings

While `CTAsPerCGA` and `WarpsPerCTA` only have one way of configuring layouts
(linearize contigously according to `shape` and `order`),
there are other ways to configure layouts for `ThreadsPerWarp` and `SizePerThread`.
Below are other layout encodings that specifies different ways of distributing `ThreadsPerWarp` and `SizePerThread`.

- Shared Layout Encoding
- MMA Layout Encoding
- Dot Operand Layout Encoding

The 3 layout encodings are related to tensor cores, which are beyond the scope of this analysis.

Finally, slice layout encoding is a layout encoding that will be used in the
[thread locality optimization pass](#optimize-thread-locality-pass).


## Axis Information

Axis information is the class `AxisInfo` in `include/triton/Analysis/AxisInfo.h`.
`AxisInfo` quantifies the properties of a layout using _lattice theory_.
Specifically, an axis information includes **contiguity**, **divisibility**, and **constancy**.

### Contiguity

A contiguity value represents the length of the shortest contiguous sequence in an array.
For example,
```python
array = [12, 13, 14, 15, 18, 19]
Contiguous sequences = { [12, 13, 14, 15], [18, 19] }
Contiguity value = 2
```

### Divisibility

A divsibility value represents the greatest common divisor of
all the first elements of the contiguous sequences in an array.
For example,
```python
array = [10, 11, 12, 13, 18, 19, 20, 21]
Contiguous sequences = { [10, 11, 12, 13], [18, 19, 20, 21] }
First elements = { 10, 18 }
Divisibility = 2
```

### Constancy

A constancy value represents the length of the shortest constant sequence in an array.
For example,
```python
array = [8, 8, 8, 8, 12, 12]
Constant sequences = { [8, 8, 8, 8], [12, 12] }
Constancy = 2
```

### Constraints

- All 3 properties are defined for every dimension in an array. That is,
  ```python
  array = [
      [12, 13, 14, 15, 18, 19],
      [22, 23, 24, 25, 28, 29]
  ]
  Contiguity = [1, 2]
  ```
- The values of all 3 properties are constrained to powers of 2.


## Optimize Thread Locality Pass

This pass optimizes the data layout for reduction operations by lowering the synchronization cost between warps in an SM.
Specifically, we recalculate the blocked layout encoding and create a new view of the data tensor.

### Get Thread Locality Optimized Encoding

Implemented in `getThreadLocalityOptimizedEncoding`, the new layout encoding is as follows:

- We first append an additional dimension to every hierarchical level to denote the reduction dimension, making every level 3-dimensional.
- For `sizePerThread`, we transpose the reduction dimension with the last one.
- For axis iteration order of threads `order`, we prepend the reduction dimension, making it the fastest changing axis.

### Get Thread Locality Optimized Shape

We also create a _view_ (abstract shape) of the data tensor shape with function `getThreadLocalityOptimizedShape`
to optimize for thread locality.
To leverage thread locality, we split the reduction dimension size into 2: **the number of elements per thread** $E$ and **the number of threads** $N$.
The dimension $E$ is appended to the tensor shape.
For example,
```python
tensor.shape = [4, 8]
tensor.elemsPerThread = [4, 4]
Reduce dimension = 1

Number of threads = 8 / 4 = 2
tensor.view = [4, 2, 4]
```
Interestingly, the code uses integer division, so the number of elements would change if the reduction dimension size is not divisible by $E$.
I suspect that the tensor shape and $E$ both have limitations in what value they can take on, so it is always divisible.


## Coalesce Pass

The coalesce pass transforms the layout of operands into coalesced memory layout.
Specifically, the pass recomputes `sizePerThread` to maximize memory access efficiency.
The core algorithm is implemented in function `setCoalescedEncoding`.

### AxisInfo

Implemented as a lambda function `queryAxisInfo`, the first step is to get the axis information of the data layout.
We first construct the 3 properties, each of them being a vector the size of the data tensor rank intialized with 1.
We then calculate the contiguity and divisibility of the `i`-th dimension, where `i` is the fastest changing axis (`order[0]`).

- Contiguity: The `i`-th dimension contiguity is set to `threadsPerCTA.shape[i]`,
- Divisibility: The `i`-th dimension divisibility is set to `128 / B`, where `B` is the number of bytes per element.

## Number of Elements Per Thread

We then recalculate the number elements of per thread in the `i`-th dimension with the axis information (`getNumElementPerThread`).
The optimal number of elements per thread is the smallest among the following:

- Maximum contiguous thread ID assignments: This is the contiguity value.
- Maximum multiple of elements: This is the divisibility value divided by the number of bytes per element.
- Maximum number of elements in a vectorized store op: 128 bits divided by the element bitwidth.

We then recalculate the rest of the layout encoding with this new value.


As of writing, **I still struggle understanding what divisibiliy represents.**  
The divisibility in `queryAxisInfo` is actually `16 * 8 / B`, but the comments in the code says the value is set to 16.  
What is more confusing is that `getNumElementPerThread` uses divisibility as
**number of bytes in a multiple**, whereas in `queryAxisInfo` `16 * 8 / B` seems to imply the unit to be **number of elements**.  
Finally, I could not see the connection between those 2 values and the original definition.


## Lowering Python Front-end code to GPU IR

I sifted through the Triton's Python front-end code and hacked up a lowering pipeline based on 2 files:

- `python/triton/compiler/compiler.py`
- `python/triton/tools/compile.py`

I implemented it in a jupyter notebook `triton-dump-ir.ipynb` and generated `compile_test_ttgir.ll`.
The IR code is fully commented. See the code for more explanation.
