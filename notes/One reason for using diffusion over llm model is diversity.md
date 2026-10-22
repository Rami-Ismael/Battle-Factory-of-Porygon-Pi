# Writing

## Hook of the paragraph

1. My friends ask me, “Why are you not using a large language model? Do you not know that large language models are the future?” I personally use large language models for everyday tasks. Here is why.

## Rest of the body


1. The first of many reason is diversity. In llm training, pretraining stage model given a query talked like a wall of text you see in math section of wikipedia. The RLHF take human feedback by improving the model to be fun to talked too and a bit chatty.In a controlled study in writing, the model that went under RLHF has reduced content compared to the base pretrained model [2]. This is called mode collapse [1]. The modal collapse is where you have wide probably distribution is narrow down to a few outputs.
# Evidence to cite

1. Kirk, Mediratta, Nalmpantis, Luketina, Hambro, Grefenstette, Raileanu — _Understanding the Effects of RLHF on LLM Generalisation and Diversity_, **ICLR 2024**. 🟢 [arXiv 2310.06452](https://arxiv.org/abs/2310.06452) · [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/5a68d05006d5b05dd9463dd9c0219db0-Paper-Conference.pdf). Reinforcement learning from human feedback "substantially reduces output diversity" both per-input and across-input — the authors name it mode collapse — and they frame it as an inherent generalisation↔diversity tradeoff.
2. - Padmakumar & He — _Does Writing with Language Models Reduce Content Diversity?_, **ICLR 2024**. 🟢 [arXiv 2309.05196](https://arxiv.org/abs/2309.05196). Controlled experiment: the feedback-tuned model produced a statistically significant diversity reduction; **the base model did not**. That nuance matters — the defect is instruction tuning, not language models as such.