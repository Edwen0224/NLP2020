# word embedding
```
給出一個文檔，
文檔就是一個單詞序列比如 “A B A C B F G”, 
希望對文檔中每個不同的單詞都得到一個對應的向量(往往是低維向量)表示。

比如，對於這樣的“A B A C B F G”的一個序列，也許我們最後能得到：
A對應的向量為[0.1 0.6 -0.5]，B對應的向量為[-0.2 0.9 0.7]  （此處的數值只用於示意）
之所以希望把每個單詞變成一個向量，目的還是為了方便計算，
比如“求單詞A的同義詞”，就可以通過“求與單詞A在cos距離下最相似的向量”來做到。

作者：li Eta
連結：https://www.zhihu.com/question/32275069/answer/61059440
```
# word embedding前世今生
```
詞的分散式表示distributed representation
傳統的獨熱表示（ one-hot representation）僅僅將詞符號化，不包含任何語義資訊。
如何將語義融入到詞表示中？
Harris 在 1954 年提出的分佈假說（ distributional hypothesis）為這一設想提供了理論基礎：
上下文相似的詞，其語義也相似。

Firth 在 1957 年對分佈假說進行了進一步闡述和明確：
詞的語義由其上下文決定（ a word is characterized by thecompany it keeps）。

到目前為止，基於分佈假說的詞表示方法，
根據建模的不同，主要可以分為三類：
基於矩陣的分佈表示、
基於聚類的分佈表示和
基於神經網路的分佈表示。

儘管這些不同的分佈表示方法使用了不同的技術手段獲取詞表示，
但由於這些方法均基於分佈假說，
它們的核心思想也都由兩部分組成：
一、選擇一種方式描述上下文；
二、選擇一種模型刻畫某個詞（下文稱“目標詞”）與其上下文之間的關係。

作者：Scofield
連結：https://www.zhihu.com/question/32275069/answer/301428835
```

```
2000年 -- 開始發展

2001年， Bengio 等人正式提出神經網路語言模型（ Neural Network Language Model ，NNLM），
該模型在學習語言模型的同時，也得到了詞向量。
所以請注意一點：詞向量可以認為是神經網路訓練語言模型的副產品。

2003年 -- 出現了相關系統敘述此方法的著作："A Neural Probabilistic Language Model". 
Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin. 
A neural probabilistic language model.
JMLR, 3:1137–1155, 2003.

A Survey on Neural Network Language Models
Kun Jing, Jungang Xu
https://arxiv.org/abs/1906.03591

2009年 -- 高維資料的降維方法 
Roweis, S. T., & Saul, L. K. (2000). 
Nonlinear Dimensionality Reduction by Locally Linear Embedding. Science, 290(5500). 
Retrieved from http://science.sciencemag.org/content/290/5500/2323

神經網路語言模型  https://www.zhihu.com/question/32275069
a) Neural Network Language Model ，NNLM
b) Log-Bilinear Language Model， LBL
c) Recurrent Neural Network based Language Model，RNNLM
d) Collobert 和 Weston 在2008 年提出的 C&W 模型
e) Mikolov 等人提出了 CBOW（ Continuous Bagof-Words）和 Skip-gram 模型


2013年 -- word2dec出現。這是由Tomas Mikolov帶領的穀歌團隊編寫的toolkit
T. Mikolov and G. Zweig. 
Context dependent recurrent neural network language model.
In IEEE SLT, pages 234–239, 2012.

T. Mikolov, M. Karafiat, L. Burget, ´ J. Cernocky, and S. Khudanpur. 
Recurrent neural network based language model. 
In INTERSPEECH, pages 1045–1048, 2010.

T. Mikolov, S. Kombrink, L. Burget, J. Cernocky, and S. Khudanpur. 
Extensions of recurrent ´neural network language model. 
In Proc. of IEEE ICASSP, pages 5528–5531, 2011.

T. Mikolov, S. Kombrink, A. Deoras, and L. Burget. 
RNNLM - Recurrent Neural Network Language Modeling Toolkit. 
In IEEE ASRU, page 4, 2011.

T. Mikolov, I. Sutskever, A. Deoras, H. Le, and S. Kombrink. 
Subword Language Modeling With Neural Networks. 2012.

T. Mikolov, K. Chen, G. Corrado, and J. Dean. 
Efficient estimation of word representations in vector space. 
CoRR, abs/1301.3781, 2013




2017年 -- 機器可以學到語言中的深意(可能算不上重要進展，但至少是篇有意思的文章)。
Caliskan-islam, A., Bryson, J. J., & Narayanan, A. (2017). 
Semantics derived automatically from language corpora necessarily contain human biases. 
Science, 356(April), 183–186.

Hinton就已經提出了distributed representation的概念“Learning distributed representations of concepts”
(只不過不是用在word embedding上面)

bengio的paper“Neural probabilistic language models”

word embedding方法就是先從文本中為每個單詞構造一組features，
然後對這組feature做distributed representations
，哈哈，相比于傳統的distributed representations，區別就是多了一步(先從文檔中為每個單詞構造一組feature)。

Tomas Mikolov在Google的時候發的這兩篇paper：
“Efficient Estimation of Word Representations in Vector Space”
“Distributed Representations of Words and Phrases and their Compositionality”
這兩篇paper中提出了一個word2vec的工具包，裡面包含了幾種word embedding的方法，
這些方法有兩個特點。一個特點是速度快，另一個特點是得到的embedding vectors具備analogy性質。
analogy性質類似於“A-B=C-D”這樣的結構，
舉例說明：“北京-中國 = 巴黎-法國”。
Tomas Mikolov認為具備這樣的性質，則說明得到的embedding vectors性質非常好，能夠model到語義。

這兩篇paper是2013年的工作，至今(2015.8)，這兩篇paper的引用量早已經超好幾百，足以看出其影響力很大。
當然，word embedding的方案還有很多，
常見的word embedding的方法有:
1. Distributed Representations of Words and Phrases and their Compositionality
2. Efficient Estimation of Word Representations in Vector Space
3. GloVe Global Vectors forWord Representation
4. Neural probabilistic language models
5. Natural language processing (almost) from scratch
6. Learning word embeddings efficiently with noise contrastive estimation
7. A scalable hierarchical distributed language model
8. Three new graphical models for statistical language modelling
9. Improving word representations via global context and multiple word prototypes

word2vec中的模型至今(2015.8)還是存在不少未解之謎，因此就有不少papers嘗試去解釋其中一些謎團，
或者建立其與其他模型之間的聯繫，下面是paper list
1. Neural Word Embeddings as Implicit Matrix Factorization
2. Linguistic Regularities in Sparse and Explicit Word Representation
3. Random Walks on Context Spaces Towards an Explanation of the Mysteries of Semantic Word Embeddings
4. word2vec Explained Deriving Mikolov et al.’s Negative Sampling Word Embedding Method
5. Linking GloVe with word2vec
6. Word Embedding Revisited: 
A New Representation Learning and Explicit Matrix Factorization Perspective
```
