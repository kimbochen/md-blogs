# Tweets

Putting my new Tweets here in case Twitter really disappeared.


## Würstchen

2024-02-13 21:25

Never really understood diffusion models well, but the paper caught my eye because of efficiency.
I'll do my best to understand the paper
https://openreview.net/forum?id=gU58d5QeGv

### What are the efficiency gains?
SD 2.1 vs. Würstchen v2
Training cost: 200K GPU hours -> 25K (!)
Inference time: 20s -> 9s @ batch size 8

### How did they manage to attain them?
LDMs operate on text-conditioned low-dim image latents
Wü. operates on text-conditioned lower-dim image latents, which is then upscaled by an intermediate latent image decoder.
Reminds me of progressive IR lowering in compilers.

### What are the efficiency gains?
SD 2.1 vs. Würstchen v2
Training cost: 200K GPU hours -> 25K (!)
Inference time: 20s -> 9s @ batch size 8

### How did they manage to attain them?
LDMs operate on text-conditioned low-dim image latents
Wü. operates on text-conditioned lower-dim image latents, which is then upscaled by an intermediate latent image decoder.
Reminds me of progressive IR lowering in compilers.

### How is training an additional model saving resources?
It enables the LDM to operate on extremely low-dim latents, reducing the model size.

### Model performance?
Seems to consistently outperform SD 2.1


## Mixtral of Experts

2024-02-14 23:23

Mixtral 8x7B, The model that IMO rekindled people's interest in MoEs.
I want to learn more about MoEs, starting from this paper.
https://arxiv.org/abs/2401.04088

### How does it perform compared to other models?
- Matches/outperforms Llama 2 70B with memory usage of 47B at the cost of lower utilization
- Outperforms Llama 70B on multilingual benchmarks and uses more % of multilingual data
- Mixtral Instruct ranks 12 at [LMSys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

### What does its MoE arch look like?
- Sparse MoE: N=8 number of experts, activates and weighted with top K=2 experts.  
- Cites MegaBlocks, Expert Parallelism, and GShard.
- MegaBlocks: https://arxiv.org/abs/2211.15841

### What did the team do differently so that Mixtral works?
Not sure. In terms of hardware efficiency, the team worked with NVIDIA to integrate SMoE TensorRT-LLM.
TensorRT-LLM: https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/

### Routing analysis: How does Mixtral decide the routing of its tokens?
- No obvious domain specialization
- Experts specialize on syntax, e.g. Python keyword "self", indentation
- Consecutive token assignments, especially for last few layers

### Probability of an expert being the first or second choice?
The paper says 1 - 6/8 * 5/7 ~= 46%.  
I find this weird: Shouldn't it be 25%?
1 - P(E_i not 1st and not 2nd choice)  
= 1 - P(E_i not 1st choice) * P(E_i not 2nd choice | E_i not 1st choice)  
= 1 - 7/8 * 6/7 = 25%


## Aespa - Post-Training Quantization of Hyper-Scale Transformers

2024-02-16 19:26

A "next-level" post-training quantization method called "aespa". I see what you did there haha  
https://arxiv.org/abs/2402.08958

![](imgs/next-level.png)

### What improvements does aespa bring?
- PTQ schemes trade off model perf. with quantization time.
- aespa claims to balance accuracy & efficiency.

### What is the method design that brings the improvements?
Separately quantize Wq, Wk, Wv in attn to minimize the reconstruction error of the attn module.

### Separate quantization costs more time. How does aespa overcome this?
By approximating the quantization objective and pre-compute refactored common terms.
That said, I don't understand how the authors Taylor expanded delta A:

![](imgs/taylor-expansion-delta-a.png)

### How does it compare with other PQT schemes?
- Efficiency: ~5x faster compared with block-wise PTQ scheme BRECQ
- Performance: Outperforms OmniQuant a lot under the same precision
Not sure why they picked OmniQuant for comparison though. Maybe for its recency?

### Comparison with zero-shot quantization
- aespa is always 1 - 2 lower ppl than other methods but really shines at INT2, with 10 - 20 lower ppl
- At INT2, the ppl gap increases with the model size
- Method Z-Fold actually holds up pretty well. I wonder how much faster Z-Fold is

### End notes
- The math is slightly out of reach for me. I should really improve my matrix calculus
- I seem to lack a bit context of quantization, including initialization (AWQ, Z-Fold), weight-rounding (AdaRound, AdaQuant), etc.


## Aya

2024-02-17 14:21

Aya: Another win for open science research after OLMo and MiniCPM.  
Aya aims to improve multilingual abilities of our models through building the dataset that includes as many languages as possible.  
The dataset: https://arxiv.org/abs/2402.06619  
The model: https://arxiv.org/abs/2402.07827

### Released data
-  Aya Dataset: The human-annotated, multilingual instr. fine-tuning dataset
- Aya Collection: Superset of Aya Dataset, including templated and translated data of selected existing datasets

### How data composition impacts task perf (5.6.1)
The authors trained 3 models: more templated (TP), more translated (TR), and more human-annotated (HA) data.
- TP performs best in discriminative tasks while having more English data, indicating cross-lingual transfer for the task
- TR greatly outperforms TP on translation tasks and is preferred for open-ended gen
- HA is limited by the dataset size :(

### Some thoughts
Building datasets is probably the task that has the highest importance-to-appreciation ratio.
We keep seeing impressive models that are essentially well-known methods trained on large-scale, high-quality data.

While the open-source community is still putting up a good fight to invent great methods, building datasets seems to be less discussed.
Aya is fighting this more difficult side of battle.

My favorite part of the dataset paper is section 7 "A Participatory Approach to Research."
The social and political challenges some volunteers have to overcome to contribute to Aya gives me hope and inspires me.  
My favorite quote:
> Including these factors in our post-mortem analysis of the project is crucial to understanding both the motivation of people willing to volunteer for open-science projects, and also to understanding the data itself: its breadth, its provenance, its shortcomings, and its living history.

I am grateful to participate in the project and contribute an infinitesimal amount of data.


## Scalable Diffusion Models with Transformers

2024-02-17 16:11

Sora again showed the scalability of transformers. I read diffusion transformers, supposedly the model Sora is based on.
https://arxiv.org/abs/2212.09748

DiT replaces U-Net in a latent diffusion model with transformers, claiming scalability
- More inference compute (smaller patch, longer seq len) almost equals better performance

  ![](imgs/scale-gflops.jpg)

- Larger models w/ less training outperforms small models w/ more training

  ![](imgs/scale-train-compute.jpg)


Sora really surprised me because my impression of text-to-video was stuck at Phenaki.  
I guess I wasn't aware of all the progress in the video generation space: VideoPoet (Dec 2023) seems already pretty good.  
https://blog.research.google/2023/12/videopoet-large-language-model-for-zero.html

VideoPoet is not a diffusion model though.  
It is a transformer with multi-axis attention and uses different encoder models to tokenize text, video, and audio.
