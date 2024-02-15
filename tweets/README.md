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
