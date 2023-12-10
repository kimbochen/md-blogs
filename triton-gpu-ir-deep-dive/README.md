# Triton GPU IR Deep Dive

## Table of Contents

- [Triton Compiler Code Structure](#triton-compiler-code-structure)
  - [Triton GPU IR](#triton-gpu-ir)
  - [Other Stuff](#other-stuff)
- [Triton GPU IR Attribute Definitions](#triton-gpu-ir-attribute-definitions)
  - [Distributed Layout Encoding](#distributed-layout-encoding)
  - [Blocked Layout Encoding](#blocked-layout-encoding)
  - [Other Layout Encodings](#other-layout-encodings)
- [Axis Information](#axis-information)
- [Optimize Thread Locality Pass](#optimize-thread-locality-pass)
- [Coalesce Pass](#coalesce-pass)
- [Lowering Python Front-end code to GPU IR](#lowering-python-front-end-code-to-gpu-ir)
- [Grouped Matrix Multiplication in Triton Language](#grouped-matrix-multiplication-in-triton-language)


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

I will be detailing the `Coalesce` and the `OptimizeThreadLocality` pass. More on that later.


### Other Stuff

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

\newpage

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

The base class for a layout is `DistributedLayoutEncoding`, which is defined with a tensor $T$.  
Let the data tensor be $A$ and its index be $i$, $rank(A) = rank(T) = D, i \in \mathbb{Z}^D$

Layout `L_T[i] = { T[idx(k)] }` $\forall k$,
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

\newpage

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
(Example taken from the source code.
[Link](https://github.com/openai/triton/blob/addd94e4a8d4dc0beefd5df6d97d57d18065436a/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L444))


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


## Optimize Thread Locality Pass

## Coalesce Pass

## Lowering Python Front-end code to GPU IR

## Grouped Matrix Multiplication in Triton Language
