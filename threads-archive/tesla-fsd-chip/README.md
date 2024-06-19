# Tesla FSD Chip

Today I'm reading about Tesla's FSD chip.
First up:
[Blog Post: Inside Tesla’s Neural Processor In The FSD Chip](https://fuse.wikichip.org/news/2707/inside-teslas-neural-processor-in-the-fsd-chip/)

Tesla designed the neural network processor in-house, while using IP blocks for other logic.

Tesla uses [Verilator](https://www.veripool.org/verilator/), an open-source hardware simulator, for simulation.
Interestingly Tesla claims it to be 50x faster than commercial ones.

NPU supports 8 instructions:
- State machine: DMA Read / Write, Stop
- Linear algebra: scaling, element-wise addition
- ML ops: Conv, Deconv, Dot prod
The ML landscape now is very different from the time NPU was designed.
I wonder if the latest chip supports different ML ops such as scaled dot product attention.

It took me some time to understand how their NPU performs convolution.
- Unrolls convolution into vector dot product
- Computes 96 output activations in parallel
- Computes 96 output channels (i.e. filters) in parallel

This is pretty cool: Instead of doing inner-product-style matmul,
it does outer-product-style, where it uses the whole MAC array to do col @ row and accumulate.

![](tesla_npu_matmul.jpg)

NPU has a programmable SIMD unit that supports activation functions and normalization layers.

Tesla implemented specialized hardware for avg and max pooling.  
Performance of memory-bound ops like activation functions become critical when compute ops are accelerated due to Amdahl's Law.

Results: 21x performance (2300 FPS) at 25% power increase (72 W)  
To put into perspective, a similar-powered Jetson AGX Orin 32GB runs at ~1400 FPS for Inception-like models.

- [NVIDIA Jetson Modules](https://developer.nvidia.com/embedded/jetson-modules)
- [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)
