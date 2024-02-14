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
