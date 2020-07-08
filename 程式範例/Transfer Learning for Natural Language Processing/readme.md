# Transfer Learning for Natural Language Processing 
```
https://www.manning.com/books/transfer-learning-for-natural-language-processing

https://github.com/azunre/transfer-learning-for-nlp
```
```
PART 1 INTRODUCTION AND OVERVIEW
1 What is Transfer Learning?
2 Getting Started with Baselines

PART 2 TRANSFER LEARNING FOR NLP
3 Shallow Transfer Learning in NLP
4 Deep Transfer Learning for NLP with Recurrent Neural Networks
5 Deep Transfer Learning in NLP with Transformers
6 Deep Transfer Learning Adaptation Strategies

7 Conclusions
Appendix A - Free GPUs in the Cloud with Kaggle Kernels and Colab Notebooks
Appendix B - Using Microservices to Handle Dependencies
```

# English word vectors
```
Wiki word vectors
https://fasttext.cc/docs/en/pretrained-vectors.html
We are publishing pre-trained word vectors for 294 languages, trained on Wikipedia using fastText. 
These vectors in dimension 300 were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.

Please note that a newer version of multi-lingual word vectors are available at: Word vectors for 157 languages.


https://fasttext.cc/docs/en/english-vectors.html
```
# Ch2
## 兩大案例
### 2.1 Preprocessing Email Spam Classification Example
### 2.2 Preprocessing Movie Sentiment Classification Example Data
##
### 2.3 Generalized Linear Models
### 2.4 Decision-Tree-Based Models
### 2.5 Neural Network Models
```
In this section, we will train two representative pretrained neural network language models on the 
 two illustrative example problems we have been baselining in this chapter. The two models we will consider here are

• ELMo – Embeddings from Language Models, and
• BERT – Bidirectional Encoder Representations from Transformers.
```
```
2	2.1-2.4	Linear & Tree-based models for Email Sentiment Classification	https://www.kaggle.com/azunre/tl-for-nlp-section2-1-2-4-emails
2	2.1-2.4	Linear & Tree-based models for IMDB Movie Review Classification	https://www.kaggle.com/azunre/tl-for-nlp-section2-1-2-4-movies

2	2.5	ELMo for Email Semantic Classification	https://www.kaggle.com/azunre/tl-for-nlp-section2-5-emails-elmo
2	2.5	ELMo for IMDB Movie Review Classification	https://www.kaggle.com/azunre/tl-for-nlp-section2-5-movies-elmo

2	2.6	BERT for Email Semantic Classification	https://www.kaggle.com/azunre/tl-for-nlp-section2-6-emails-bert
2	2.6	BERT for IMDB Movie Review Classification
```

# Ch3.Shallow Transfer Learning for NLP
```
3.1 Semi-supervised Learning with Pretrained Word Embeddings
3.2 Semi-supervised Learning with Higher-Level Representations
3.3 Multi-Task Learning
3.4  Domain Adaptation
```

```
3.1	IMDB Review Classification with word2vec and FastText	https://www.kaggle.com/azunre/tl-for-nlp-section3-1-movies-word-embeddings
3	3.2	IMDB Review Classification with sent2vec	https://www.kaggle.com/azunre/tl-for-nlp-section3-2-movies-sentence-embeddings
3	3.3	Dual Task Learning of IMDB and spam detection	https://www.kaggle.com/azunre/tl-for-nlp-section3-3-multi-task-learning
3	3.4	Domain adaptation of IMDB classifier to new domain of Book Review Classification	
```
# 3-1-movies-word-embeddings


# 4 Deep Transfer Learning for NLP with Recurrent Neural Networks
```
4.1 Preprocessing Tabular Column Type Classification Data範例一
4.2 Preprocessing Fact Checking Example Data  範例二:假新聞偵測

4.3 Semantic Inference for the Modeling of Ontologies (SIMOn)
4.4 Embeddings from Language Models (ELMo)
4.5 Universal Language Model Fine-Tuning (ULMFiT)
```
```
4.1 & 4.3	Using SIMOn for column data type classification on baseball and BC library OpenML datasets	
            https://www.kaggle.com/azunre/tl-for-nlp-section4-1-4-3-column-type-classifier
4	4.2 & 4.4	Using ELMo for "fake news" detection/classification
```
