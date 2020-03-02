# 2020
```
UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training
Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Songhao Piao, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon
(Submitted on 28 Feb 2020)

https://arxiv.org/abs/2002.12804

We propose to pre-train a unified language model for both autoencoding 
and partially autoregressive language modeling tasks using a novel training procedure, 
referred to as a pseudo-masked language model (PMLM). 

Given an input text with masked tokens, 
we rely on conventional masks to learn inter-relations between corrupted tokens and context via autoencoding, 
and pseudo masks to learn intra-relations between masked spans via partially autoregressive modeling. 

With well-designed position embeddings and self-attention masks, 
the context encodings are reused to avoid redundant computation. 

Moreover, conventional masks used for autoencoding provide global masking information, 
so that all the position embeddings are accessible in partially autoregressive language modeling. 

In addition, the two tasks pre-train a unified language model as a bidirectional encoder 
and a sequence-to-sequence decoder, respectively. 

Our experiments show that the unified language models pre-trained using PMLM achieve new state-of-the-art results 
on a wide range of natural language understanding and generation tasks across several widely used benchmarks.
```

#
```

```

#
```

```


# 2017
```
Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
(Submitted on 12 Jun 2017 (v1), last revised 6 Dec 2017 (this version, v5))
https://arxiv.org/abs/1706.03762

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks 
in an encoder-decoder configuration. 

The best performing models also connect the encoder and decoder through an attention mechanism. 

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, 
dispensing with recurrence and convolutions entirely. 

Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable 
and requiring significantly less time to train. 

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, 
improving over the existing best results, including ensembles by over 2 BLEU. 
On the WMT 2014 English-to-French translation task, 
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, 
a small fraction of the training costs of the best models from the literature. 

We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing 
both with large and limited training data.
```
```

```
