#
```
深度學習的發展歷史
http://www2.econ.iastate.edu/tesfatsi/DeepLearningInNeuralNetworksOverview.JSchmidhuber2015.pdf
```
# 學習主題
### 基礎技術:RNN LSTM GRU
```
colah's blog
Understanding LSTM Networks Posted on August 27, 2015
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://blog.csdn.net/menc15/article/details/71271566
https://blog.csdn.net/Jerr__y/article/details/58598296
```
```
```
### 後續發展
```

```
### Transfer Learning與Pre-Trained Model
```
Google Transformer(2017)
BERT [Devlin et al., 2018], 

GPT [Radford et al., 2019], XLNet [Yang et al., 2019],

ERNIE(Enhanced Representation through kNowledge IntEgration)[2019]

自然語言處理標竿2019最後一波測試，百度打敗微軟、Google
百度的預訓練語言模型ERNIE，在GLUE平台2019年底最後一次16項包含中英文的自然語言處理測試中拿下第一，
表現優於Google、微軟和卡內基美隆大學
文/林妍溱 | 2020-01-01發表
https://www.ithome.com.tw/news/135127
```
### 
```
benchmarks: GLUE [Wang et al., 2018] and RACE [Lai et al., 2017] 
GLUE leaderboard: https://gluebenchmark.com/leaderboard/, 
RACE leaderboard: http://www.qizhexie.
com/data/RACE_leaderboard.html
```
```
GLUE（General Language Understanding Evaluation）是知名的自然語言理解（NLU）多任務標竿測試和分析平台，
包含9項測試，像是聽取句子中的人名和組織名，或是聽句子中的同音異形字等等。

```
```
Fake News Challenge Stage 1 (FNC-1)
http://www.fakenewschallenge.org/

https://github.com/FakeNewsChallenge

https://arxiv.org/pdf/1910.14353.pdf

```


### 2019_RNN實戰主題
```
[1]TextClassification文本分類
   IMDb-Movie-Review IMDb網路電影影評資料集 Sentiment Analysis on IMDb
[2]TextGeneration文本生成:作詞機器人
[3]資安專題:垃圾短信(分類)偵測
```

# NLP 學習資源
```
https://github.com/graykode/nlp-tutorial
```
```
http://web.stanford.edu/class/cs224u/index.html
https://github.com/cgpotts/cs224u/

http://cs224d.stanford.edu/
```
# NLP
```
NLP，全名Natural Language Processing（自然語言處理），是一門集電腦科學，人工智慧，語言學三者於一身的交叉性學科。
她的終極研究目標是讓電腦能夠處理甚至是“理解”人類的自然語言，進而説明人類解決一些現實生活中遇到的實際問題。
這裡的語言“理解”是一個很抽象也很哲學的概念。
在NLP中，我們將對語言的“理解”定義為是學習一個能夠解決具體問題的複雜函數的過程。

對NLP的研究通常在5個Level上進行：
語音／文本分析：包括語言識別技術、OCR技術、分詞技術等
詞形分析：例如分析一個word的前尾碼、詞根等
語法分析：從語法結構上分析一個句子的構成
語義分析：理解一個句子或單詞的語義
篇章分析：理解一段篇章的含義
注意，雖然這5個Level在研究的物件和難度上是一種遞進的關係，但對這5個Level的研究並不一定是連續的——例如，
我們可以跳過對一個句子語法結構的分析而直接去理解句子的語義。

接下來簡單列舉一些NLP技術的應用，以讓大家對NLP能夠處理什麼樣的問題有一個感性的認識：

簡單的任務：拼寫檢查，關鍵字檢索，同義詞檢索等
複雜一點的任務：資訊提取（比如從網頁中提取價格，產地，公司名等資訊），情感分析，文本分類等
更複雜的任務：機器翻譯，人機對話，QA系統

https://www.cnblogs.com/iloveai/p/cs224d-lecture1-note.html
```

# NLP應用
```
文字朗讀（Text to speech）/語音合成（Speech synthesis）
語音識別（Speech recognition）
中文自動分詞（Chinese word segmentation）
詞性標註（Part-of-speech tagging）
句法分析（Parsing）
自然語言生成（Natural language generation）
文字分類（Text categorization）
資訊檢索（Information retrieval）
資訊抽取（Information extraction）
文字校對（Text-proofing）
問答系統（Question answering）
機器翻譯（Machine translation）
自動摘要（Automatic summarization）
文字蘊涵（Textual entailment）
命名實體辨識（Named entity recognition）
```
# NLP工具與模組
```

```
### NLP工具與模組:Gensim
```
Gensim是一個開源庫，使用現代統計機器學習來進行無監督的主題建模和自然語言處理。
Gensim是用Python和Cython實現的。
Gensim旨在使用數據流和增量在線算法處理大型文本集合，這使其有別於僅針對內存中處理的大多數其他機器學習軟件包。
https://radimrehurek.com/gensim/
```
```
15分钟入门Gensim
https://zhuanlan.zhihu.com/p/37175253
https://www.cnblogs.com/iloveai/p/gensim_tutorial.html

Gensim进阶教程：训练word2vec与doc2vec模型
https://www.cnblogs.com/iloveai/p/gensim_tutorial2.html
```
```
使用gensim訓練維琪百科中文語料wordvec模型
https://www.jianshu.com/p/e2d13d058ac6
```
### 

```

```

# NLP應用1:TextClassification 文本分類
```
TextClassification 文本分類的意義
二元分類:分成好與惡
多元分類:分成不同類型的文章或喜好度
```
```
TextClassification 文本分類
IMDb-Movie-Review IMDb網路電影影評資料集 Sentiment Analysis on IMDb
   
Sentiment Analysis on IMDb
https://paperswithcode.com/sota/sentiment-analysis-on-imdb

情緒分析sentiment analysis
Sentiment analysis (opinion mining or emotion AI) 
https://en.wikipedia.org/wiki/Sentiment_analysis
很多線上社群網站會蒐集使用者的資料，並且分析使用者行為，
像是知名的Facebook在前幾年開始做「情緒分析(sentiment analysis)」，
情緒分析是以文字分析、自然語言處理NLP的方法，找出使用者的評價、情緒，進而預測出使用者行為來進行商業決策，
像這樣一連串利用情緒分析帶來的商業價值是相當可觀的。
```
## NLP應用1:TextClassification 文本分類 技術主題
```
TF-IDF與機器學習實現
https://blog.csdn.net/crazy_scott/article/details/80830399

使用TensorFlow Estimators
https://ruder.io/text-classification-tensorflow-estimators/

使用TF-IDF and ELMo
https://www.kaggle.com/saikumar587/imdb-text-classification-tf-idf-and-elmo

使用BERT (2018)
Sentiment Analysis of IMDb Movie Reviews Using BERT
BERT Text Classification in 3 Lines of Code Using Keras
https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358


使用XLNet(2019)
XLNet: Generalized Autoregressive Pretraining for Language Understanding
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
(Submitted on 19 Jun 2019)
https://arxiv.org/abs/1906.08237
https://github.com/zihangdai/xlnet

範例程式
https://colab.research.google.com/github/zihangdai/xlnet/blob/master/notebooks/colab_imdb_gpu.ipynb
```



# NLP應用2:Text Generation文本生成
```
TextGeneration文本生成:作詞機器人

範例程式
參考資料
sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python
https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python

Hands-On-Deep-Learning-Algorithms-with-Python/
04. Generating Song Lyrics Using RNN/4.06 Generating Song Lyrics Using RNN.ipynb
```


#
```

```


#
```

```

```
