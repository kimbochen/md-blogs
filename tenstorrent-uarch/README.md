# Tenstorrent Micro-architecture

## Tenstorrent HW Architecture Overview

Watching a presentation about Tenstorrent's HW architecture.
[Presentation Link](https://www.youtube.com/watch?v=Id3enIOAY2Q)  
A live thread:

Jawbridge, their first test chip  
4 full-fledged CPU core (!)  
1 TFLOP / s, 6MB SRAM, 1.5W

Grayskull  
126 Tensix cores  
92 TFLOP / s, 120 MB SRAM, 65W

Wormhole  
Ethernet chip-to-chip, chip-to-switch connectivity  
~2x perf and power of Grayskull

Wormhole chip will be a 75W PCIE card and a system chip

Nebula - 4U server: 32 (8 * 4) Wormholes, 384 GB DRAM  
Galaxy: 8 Nebulas, Supercomputer topology-2D mesh (2 * 4)  
Rack-to-Rack: 2 * N through ethernet  
Wormhole has built-in switch, no need extra switches  
Also supports datacenter topology for redundancy & multi-tenancy

Presented by Drago Ignjatovic (VP of HW) and Davor Capalija (Fellow)

Free model designers from scaling hierarchy with a compiler that performs place-and-route on a core mesh

DistBelief / Parameter server architecture / Limited by SRAM capacity of a single node due to data parallelism

Tenstorrent u-arch: Place-and-route of Mini-tensor graphs  
Mini-tensor: tensor shards in model / data parallelism  
u-arch natively supports running MP / DP with interconnect

Thoughts:  
Does TT's compiler work more like GSPMD or ZeRO?  
Scaling topology limited by bandwidth like Tesla Dojo?


## Tenstorrent Blackhole Architecture

Today I'm reading a blog post about Tenstorrent's latest chip Blackhole by Semianalysis:  
[Link](https://www.semianalysis.com/p/tenstorrent-blackhole-grendel-and)

Paraphrasing the author's words: Models have evolved to leverage GPUs, and GPUs have evolved to accelerate successful models.  
The feedback loop leads algorithm development towards a local maximum of algo-hardware combination.  
That was a succinct way of explaining hardware lottery.

Tenstorrent's vision of future ML models are mixture-of-experts-style models with potentially even more complex routing.  
This is in line with Google's vision (PaLM).  
Given the success of Mixtral 8x7B, mixture of experts model definitely looks promising.  
I wonder how fast Tenstorrent's chip can run Mixtral 8x7B.

I disagree that more conditional routing is good for GPUs.  
Conditional routing would cause low effective memory utilization because GPUs load all the weights into VRAM. GPUs are also bad at conditionals.  
It causes warp divergence and tanks the performance.  
In fact, I'd argue that it's exactly because GPUs are not good at conditional routing, mixture-of-expert models had trouble going mainstream.

Tenstorrent aims to offer end-to-end solution that includes data preprocessing, something often done on CPUs.

Tenstorrent's architecture is designed for sparsity and conditional routing.  
The built-in network capability in each core enables this.

Tenstorrent next step is to build a standalone computer, Blackhole being the first one, and Grendel being the next one with in-house designed CPU.

One interesting aspect about the CPU architecture is that they share the same routing system with the Tensix cores,
removing the complications of the host and device model.

Integrating non-differentiable, general-purpose workloads into ML computation graphs,
e.g. mixing database querying nodes into a forward-and-backward graph.  
Interesting!

Tenstorrent tries to solve the memory-bounded issue in inference with hardware they call pipes, persistent communication channels between cores.  
The compiler overlays pipes with computation to hide memory latency.  
The unique approach also supports their claim that operator fusion in kernels are not needed in compiler optimizations.  
Specialized HW with simple SW vs. Traditional HW with optimized SW, I wonder which one would win?
