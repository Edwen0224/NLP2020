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
### AUTOML since 2018
```
自動機器學習（AutoML）和神經架構搜索（NAS）
原文網址：https://kknews.cc/tech/vo4k4zl.html

Neural architecture search (NAS) 自動神經架構搜索
https://zhuanlan.zhihu.com/p/42924585

https://kknews.cc/zh-tw/tech/opjpgy5.html

https://github.com/D-X-Y/Awesome-NAS
```
```
TF之AutoML之AdaNet框架

相關論文：《 AdaNet: Adaptive Structural Learning of Artificial Neural Networks》
論文地址：http://proceedings.mlr.press/v70/cortes17a/cortes17a.pdf
Github 專案地址：https://github.com/tensorflow/adanet
教程 notebook：https://github.com/tensorflow/adanet/tree/v0.1.0/adanet/examples/tutorials
```
```
自动机器学习工具全景图：精选22种框架，解放炼丹师
关注前沿科技 量子位 2018-08-22
Automatic Machine Learning (AutoML) Landscape Survey
https://medium.com/georgian-impact-blog/automatic-machine-learning-aml-landscape-survey-f75c3ae3bbf2
```
```
比谷歌AutoML快110倍，全流程自動機器學習平台應該是這樣的
by 行動貝果 MoBagel Inc.
```
#### NASNet search space(2017)
```
Learning Transferable Architectures for Scalable Image Recognition
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le
(Submitted on 21 Jul 2017 (v1), last revised 11 Apr 2018 (this version, v4))
https://arxiv.org/abs/1707.07012
```
#### PNAS(2017)
```
Progressive Neural Architecture Search
Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy
(Submitted on 2 Dec 2017 (v1), last revised 26 Jul 2018 (this version, v3))
https://arxiv.org/abs/1712.00559
```
#### ENAS(2018)
```
Efficient Neural Architecture Search via Parameter Sharing
Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean
(Submitted on 9 Feb 2018 (v1), last revised 12 Feb 2018 (this version, v2))
https://arxiv.org/abs/1802.03268
```
### Auto-Keras(2018)
```
Auto-Keras: An Efficient Neural Architecture Search System
Haifeng Jin, Qingquan Song, Xia Hu
(Submitted on 27 Jun 2018 (v1), last revised 26 Mar 2019 (this version, v3))
https://arxiv.org/abs/1806.10282

https://autokeras.com/

https://autokeras.com/docker/

https://www.itread01.com/content/1542451324.html
```

```
Auto-Keras Docker

docker pull garawalid/autokeras:latest

docker run -it --shm-size 2G garawalid/autokeras /bin/bash
```
```
pip3 install autokeras
AutoKeras is only compatible with Python 3 and TensorFlow >= 2.1.0 
```
```
import numpy as np
import tensorflow as tf

import autokeras as ak


def imdb_raw():
    max_features = 20000
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features,
        index_from=index_offset)
    x_train = x_train
    y_train = y_train.reshape(-1, 1)
    x_test = x_test
    y_test = y_test.reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


# Prepare the data.
(x_train, y_train), (x_test, y_test) = imdb_raw()
print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # <START> this film was just brilliant casting <UNK>

# Initialize the TextClassifier
clf = ak.TextClassifier(max_trials=3)
# Search for the best model.
clf.fit(x_train, y_train)
# Evaluate on the testing data.
print('Accuracy: {accuracy}'.format(clf.evaluate(x_test, y_test)))
```

### GOOGLE Cloud AutoML(2018)| iKala Cloud
```
【Next舊金山直擊】李飛飛：AutoML Vision進入公開測試，同時推出兩項AutoML新服務：自然語言理解和翻譯自動客製建模服務
Google宣布推出了兩項AutoML新服務，
這是一個可以客製出自己的自然語言理解模型的新服務AutoML Natural Language。
另外還有一項AutoML Translation，可以客製出自己的翻譯機器學習模型。
文/王宏仁 | 2018-07-25發表 https://www.ithome.com.tw/news/124750

https://cloud.google.com/automl/docs/?hl=zh-tw
```
```
https://cloud.google.com/automl/

Cloud AutoML
即使您對機器學習所知不多，也可透過 Cloud AutoML 運用機器學習的強大功能。
您可以使用 AutoML，以 Google 機器學習功能為基礎，建立與您的業務需求完美契合的自訂機器學習模型，然後將這些模型整合到您的應用程式和網站中。

您可以使用下列 AutoML 產品建立自訂機器學習模型：

Cloud AutoML Natural Language
AutoML Natural Language 分類
您可以運用 AutoML Natural Language 分類功能訓練專屬的自訂機器學習模型，按照您定義的標籤來分類文件。

AutoML Natural Language 實體擷取
您可以運用 AutoML Natural Language 實體擷取功能訓練專屬的自訂機器層級模型，藉此辨識英語文字中的一組自訂實體。

AutoML Natural Language 情緒分析
您可以運用 AutoML Natural Language 情緒分析功能訓練專屬的自訂機器層級模型，以便分析英語文字中的語氣。


Cloud AutoML Tables
AutoML Tables
AutoML Tables 讓您的整個團隊都能自動建構及部署最先進的結構化資料機器學習模型，大幅提升作業速度，並擴大處理規模。


Cloud AutoML Translation
AutoML Translation
您可以運用 AutoML Translation 建立專屬的自訂翻譯模型，使翻譯查詢作業可傳回網域特定的結果。

Cloud AutoML Video Intelligence
Cloud AutoML Video Intelligence 分類
您可以運用 AutoML Video Intelligence 分類功能訓練機器學習模型，按照您自行定義的標籤來分類影片中的鏡頭和片段。

Cloud AutoML Video Intelligence 物件追蹤
您可以運用 Cloud AutoML Video Intelligence 物件追蹤功能訓練機器學習模型，追蹤影片中特定物件的移動情形。

Cloud AutoML Vision
AutoML Vision 分類
您可以運用 AutoML Vision 分類功能訓練專屬的自訂機器學習模型，按照您定義的標籤來分類圖片。

AutoML Vision Edge
您可以運用 AutoML Vision 訓練專屬的自訂機器學習模型，按照您定義的標籤來分類圖片。

AutoML Vision 物件偵測
您可以運用 AutoML Vision 物件偵測功能訓練專屬的自訂機器層級模型，
藉此偵測與擷取多個物件，並提供各個物件的相關資訊，包括物件在圖片中的位置。
```
```
Google 機器學習三大服務：AutoML, Cloud ML Engine, ML API 介紹與比較
https://blog.gcp.expert/google-cloud-automl-ml-engine-ml-api/
```
```
Google 資訊安全白皮書：Google Infrastructure Security(2017)
https://cloud.google.com/security/infrastructure/design

https://blog.gcp.expert/google-infrastructure-security/
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
Stanford 團隊出的 CoreNLP
https://stanfordnlp.github.io/CoreNLP/index.html

Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. 
The Stanford CoreNLP Natural Language Processing Toolkit 
In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.
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
