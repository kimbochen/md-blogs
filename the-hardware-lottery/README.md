# The Hardware Lottery

[Blog post Link](https://hardwarelottery.github.io/)

## The Idea that Wins the Lottery
A research idea wins not because it is universally superior but because it suits the available software and hardware.
In a sense, the research idea won the lottery of being compatible with the hardware and software.


## The Persistence of the Hardware Lottery
Ths issue is that building hardware is time-consuming and resource-intensive,
which means that one needs to be heavily icentivized to build hardware for a research idea.
This makes it difficult for scientists to try out novel ideas that are off the beaten path and prove the ideas' worth,
which is a chicken-and-egg problem.


## The Current Landscape
The end of Moore's Law and the breakdown of Dennard scaling forces
engineers to turn to domain-specific accelerators in order to further improve performance.
This unfortunately exacerbates the hardware lottery problem,
making the space of algorithm that suits hardware even smaller.


## The Way Forward
The ideal way of avoiding the hardware lottery problem is to lower the cost of exploring software-hardware-algorithm combinations,
which is difficult especially for hardware design.
One interesting statement I saw is that

> Machine learning researchers do not spend much time talking about how hardware chooses which ideas succeed and which fail.

I thought about how many of the research ideas in machine learning is tied around the GPU hardware.
[Transformers](https://arxiv.org/abs/1706.03762) run fast on GPUs,
[Flash Attention](https://arxiv.org/abs/2205.14135) tries to work around the memory issue of GPUs, ... etc.
It is important to improve the comunication betweeen algorithm researchers and hardware designers.


## Notes
The paper introduced many previously failed attempts of massively parallel hardware, including Connection Machine and
[Fifth Generation Computer](https://www.nytimes.com/1992/06/05/business/fifth-generation-became-japan-s-lost-generation.html).
It is really interesting to learn about the failed attempts and the context.
