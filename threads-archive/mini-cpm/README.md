# Mini-CPM

Recent wins of open-source language models: OLMo by Allen AI and MiniCPM.
It's been a while since project at this scale release so many details.
A thread:

MiniCPM: A family of 2B models trained by Chinese researchers, supposedly ranking closely to Mistral 7B.  
MiniCPM to me is the tipping point of recognizing Chinese academia's capability of training models.  
ML research from China has always been less discussed in the western world until now.  
[Blog Post: MiniCPM: Unveiling the Potential of End-side Large Language Models](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)

Interesting experiments, citing µ-Transfer and Cerebras-GPT for hparam tuning, OpenAI scaling laws for batch size tuning, and their own LR scheduler.

- [µ-Transfer](https://arxiv.org/abs/2203.03466)
- [Cerebras-GPT](https://arxiv.org/abs/2304.03208)
- [OpenAI Scaling Laws](https://arxiv.org/abs/2001.08361)

I have seen a discussion somewhere about cosine LR scheduler that says cosine LRS requires training steps to be predetermined to work well.  
MiniCPM's experiments seem to agree with this.

They claim that maintaining a constant high LR and then decay for 10% steps would outperform cosine annealing.  
[Cosine LR scheduler paper](https://arxiv.org/abs/1608.03983v5)

Batch size scheduling has a similar effect as LR scheduling, but clashes with the LR schedulers they used.  
Interesting. I wonder if anyone studied batch size scheduling?

Oh my goodness the report is unbelievably detailed.  
Honestly I am quite out of touch with the latest ML training techniques, and I am really learning a lot from this report.  
Feeling grateful to the team.

Introducing high-quality data during LR annealing phase instead of during fine-tuning increases model performance.  
This matches with [Jeremy Howard's view](https://www.latent.space/p/fastai#%C2%A7replacing-fine-tuning-with-continued-pre-training)
on throwing away fine-tuning.

The model sizes are relatively small because they are aiming for edge deployment.  
While it seems impressive that the models are punching up their weights, I'm not sure how much I should read into the evaluation scores.
