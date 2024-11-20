# MLSys 2024

This blog post is my preparation for attending the MLSys 2024 conference.

- [Conference Website](https://mlsys.org/virtual/2024/index.html)
- [2024 MLSys Young Professionals Symposium](https://sites.google.com/view/mlsys24yps/home)

Day 1 of the MLSys conference is the Young Professionals Symposium.


## Day 1 Sponsor Lightning Talks

### MatX

[Intro Twitter post](https://twitter.com/MatXComputing/status/1772615554421170562)

- MatX designs hardware that maximizes performance for large (7B+) transformer models
- Prioritizes performance (I assume this means throughput) per dollar
- Focuses on scale-out performance
- MatX claims to enable researchers **train 7B-class models from scratch within a day**

MatX recently released their research-focused LLM codebase [Seqax](https://github.com/MatX-inc/seqax)
([Explanation thread from Reiner Pope](https://twitter.com/reinerpope/status/1787663349326778430)).

- Scales out to ~100 GPUs / TPUs
- Supports FSDP and tensor parallelism with a new library `shardlib`, which allows expressing sharding configs like einops
- Explicit forward and backward pass design, e.g. explicit activation checkpointing

My question: What hardware design choices that MatX makes are different from other ML hardware accelerators?


### d-Matrix

- [Website](https://www.d-matrix.ai/)

I couldn't find much details in their architecture,
but according to [this whitepaper](https://www.d-matrix.ai/wp-content/uploads/2023/10/d-Matrix-WhitePaper.pdf):
- Uses chiplet design to overcome high manufacture cost
- Has a in-memory-processing-like architecture Digital In Memory Compute (DIMC) to improve throughput

Some questions I have:
- One of chiplet design's disadvantages is the reliant of high-speed interconnect, a critical issue to LLM workloads.
  How does d-Matrix solve this issue?
- What does the software stack for programming d-Matrix hardware look like?


### SambaNova

- [My thread on SambaNova](https://twitter.com/kimbochen/status/1760423092520997203)
- [Presentation on their hardware SN30](https://www.youtube.com/watch?v=-LrOJK-tUIk)

SambaNova is one of the few ML hardware startups that are actually releasing models.  
Their Composition of Experts models are really similar to what the open-source community calls FrankenMoEs.  
Like Groq, SambaNova's hardware also touts low-latency inference (~500 tokens per second for 7B-class models).


### Databricks

Databricks ML research, formerly Mosaic ML, is known for pushing ML efficiency.  
In late March, they released [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm),
a mixture-of-experts model on par with GPT-3.5.  
They also took in [MegaBlocks](https://www.databricks.com/blog/bringing-megablocks-databricks),
the MoE implementation that powers most popular MoE models, demonstrating their commitment to improving MoE model efficiency.

My question: What is Databricks' opinion on recent different MoE configurations like DeepSeek v2?


### Together AI

Together AI has released so many cool stuff I don't know which one to read,
but I'll start with [Sequoia](https://www.together.ai/blog/sequoia).


### Snowflake

Snowflake AI research recently released Arctic in collaboration with Together AI.  

