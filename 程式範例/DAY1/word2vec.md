# word2vec
```
word2vec 是 Google 的一個開源工具，
能夠根據輸入的「詞的集合」計算出詞與詞之間的距離。

它將「字詞」轉換成「向量」形式，可以把對文本內容的處理簡化為向量空間中的向量運算，
計算出向量空間上的相似度，來表示文本語義上的相似度。
word2vec 計算的是餘弦值 (cosine)，距離範圍為 0–1 之間，值越大代表兩個詞關聯度越高。
詞向量：用 Distributed Representation 表示詞，通常也被稱為「Word Representation」或「Word Embedding」。
```
```
https://zh.wikipedia.org/wiki/Word2vec


https://easyai.tech/ai-definition/word2vec/
https://zhuanlan.zhihu.com/p/26306795
```
```
Gensim Word2Vec 簡易教學
https://www.kaggle.com/jerrykuo7727/word2vec
```
```
Word2Vec 模型中，主要有 CBOW 與 Skip-gram 兩種模型。
從直觀上來理解， 
Skip-gram 是給定輸入字詞後，來預測上下文；
CBOW 則是給定上下文，來預測輸入的字詞

http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
https://medium.com/@tengyuanchang/讓電腦聽懂人話-理解-nlp-重要技術-word2vec-的-skip-gram-模型-73d0239ad698
```

```
Efficient Estimation of Word Representations in Vector Space
https://arxiv.org/pdf/1301.3781.pdf

Distributed Representations of Words and Phrases and their Compositionality
https://arxiv.org/pdf/1310.4546.pdf
```
