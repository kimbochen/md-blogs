# Mobile V-MoEs: Scaling Down Vision Transformers via Sparse Mixture-of-Experts

[Paper Link](https://arxiv.org/abs/2309.04354)

Mixture-of-Experts models improves efficiency by activating a small subset of model weights for a given input,
or as the author puts it (which I really like), **decoupling model size from inference efficiency**.  
MoEs have seen great success in LLMs, and this paper explores MoEs under resource-constrained settings.
Specifically, the authors trained a MoE model of Vision Transformers for image recognition task.

Large scale MoE models route parts of image, i.e. image patches, to experts, and thus may activate multiple experts per image.
To reduce the amount of resource needed, the authors proposed routing **whole images** to experts.  
The author also proposed to replace end-to-end learning of the router with partially pre-defined logic.
Instead of letting the router learn the data distribution through training,
we first cluster classes into **super-classes** and train the router to identify super-classes.

Experimental results show that given the same amount of FLOPs, V-MoE outperforms its dense counterpart on every tested model configuration.
One intriguing ablation is the routing strategies.

| Routing | Input | $\Delta$ |
| :-: | :-: | :-: |
| End-to-end | image | +1.87 |
| Super-class | image | +2.72 |
| End-to-end | token | +3.15 |

This ablation showed that super-classing improved the accuracy by 0.85,
whereas token-based routing, i.e. routing patches, improved the accuracy by 1.28.
This shows that token-based routing is superior and truly takes advantage of MoEs' strength.
I believe this is a direction worth exploring for both ML algorithm and computer architecture design.

One result I hoped for is latency, since routing is quite different from how most deep learning models work and not as well supported.
Unfortunately, this paper left it for future work.
Nonetheless, running MoEs in resource-constrained setting is such an interesting idea.
