# 
```

CS224n: Natural Language Processing with Deep Learning
Stanford / Winter 2020
http://web.stanford.edu/class/cs224n/index.html
```
```
http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/word2vec%20(v2).pdf
```

# 各式資料集
```
General Language Understanding Evaluation (GLUE) benchmark 
https://gluebenchmark.com/
```
```
Penn Tree Bank (PTB) [MKM+94] dataset

LAMBADA dataset [PKL+16] 
tests the modeling of long-range dependencies in text – the model is asked to
predict the last word of sentences which require reading a paragraph of context

HellaSwag dataset [ZHB+19] involves picking the best ending to a story or set of instructions. 

StoryCloze 2016 dataset [MCH+16], which involves selecting the correct ending
sentence for five-sentence long stories.

Closed Book Question Answering
Natural Questions [KPR+19],
WebQuestions [BCFL13], and TriviaQA [JCWZ17],

Reading Comprehension
```
# language model 語言模型
```

https://zhuanlan.zhihu.com/p/42618178
```
```
語言模型（Language Model, LM）做的事情就是在給定一些詞彙的前提下， 去估計下一個詞彙出現的機率分佈
```
## wordvec(2013)
```
http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/word2vec%20(v2).pdf
```
```
http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture01-wordvecs1.pdf
```
### ELMo(2018)
```
ELMo是一種在詞向量（vector）或詞嵌入（embedding）中表示詞彙的新方法
https://arxiv.org/pdf/1802.05365.pdf
（1）ELMo能夠學習到詞彙用法的複雜性，比如語法、語義。
（2）ELMo能夠學習不同上下文情況下的詞彙多義性。

https://zhuanlan.zhihu.com/p/42618178
```

# autoregressive language model 

# 機器翻譯Machine Translation
```
Machine Translation, Seq2Seq and Attention
http://web.stanford.edu/class/cs224n/slides/cs224n-2020-lecture08-nmt.pdf
```
# seq2seq(2014)
```
Sequence to Sequence Learning with Neural Networks
Ilya Sutskever, Oriol Vinyals, Quoc V. Le
https://arxiv.org/abs/1409.321
```
```
深度學習：Seq2seq模型
其他 · 發表 2019-01-24
https://www.itread01.com/content/1548323656.html
```
```

https://easyai.tech/ai-definition/encoder-decoder-seq2seq/
```
```

https://easyai.tech/ai-definition/encoder-decoder-seq2seq/
```

```
Tensorflow 2.0 之“机器翻译”
https://zhuanlan.zhihu.com/p/1509099

使用一個非常簡單的資料集（僅有20個句子）和經典的Seq2Seq模型，應用TensorFlow2.0來訓練。

我們將通過以下步驟實現：
數據準備
沒有注意力機制的Seq2Seq
有注意力機制的Seq2Seq
```

# Attention Model
```
官方原始碼
Neural machine translation with attention
https://www.tensorflow.org/tutorials/text/nmt_with_attention
```
```
Attention 机制  不錯
https://easyai.tech/ai-definition/attention/
```
```
Attention原理和源码解析
https://zhuanlan.zhihu.com/p/43493999
```
```
C5W3L07 Attention Model Intuition
https://www.youtube.com/watch?v=SysgYptB198

C5W3L08 Attention Model
https://www.youtube.com/watch?v=quoGRI-1l0A
```
# Transformer
```
官方原始碼
Transformer model for language understanding
https://www.tensorflow.org/tutorials/text/transformer
```
```
【NLP】Transformer详解
https://zhuanlan.zhihu.com/p/44121378
```
```
Transformer Neural Networks - EXPLAINED! (Attention is all you need)
https://www.youtube.com/watch?v=TQQlZhbC5ps
```
```
LSTM is dead. Long Live Transformers!
https://www.youtube.com/watch?v=S27pHKBEp30
```
# pre-train model
```
預訓練的語言表徵經過精調後可以在眾多NLP任務中達到更好的表現。目前預訓練有兩種方法：

Feature-based：將訓練出的representation作為feature用於任務，
     從詞向量、句向量、段向量、文本向量都是這樣的。
     新的ELMo也屬於這類，但遷移後需要重新計算出輸入的表徵。
Fine-tuning：這個主要借鑒於CV
     在預訓練好的模型上加些針對任務的層，再對後幾層進行精調。
     新的ULMFit和OpenAI GPT屬於這一類。
```
```
預訓練（Pre-train）語言模型可用於自然語言理解（Natural Language Understanding）的
命名實體識別（Named Entity Recognition）、問答（Extraction-based Question Answering）、情感分析（Sentiment analysis）、
文件分類（Document Classification）、自然語言推理（Natural Language Inference）等任務。
以及自然語言生成（Natural Language Generation）的機器翻譯（Machine translation）、自動摘要（Automatic summarization）、
閱讀理解（Reading Comprehension）、資料到文本生成（Data-to-Text Generation）
```
```
State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0
https://github.com/huggingface/transformers

http://speech.ee.ntu.edu.tw/~tlkagk/courses_DLHLP20.html
```

# BERT
```
Devlin, J., Chang, M.-W., Lee, K., and Toutanova,K. (2019). 
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
arXiv:1810.04805 
```
```
官方原始碼
Fine-tuning a BERT model
https://www.tensorflow.org/official_models/fine_tuning_bert
```
```
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

TensorFlow 2 — BERT: Movie Review Sentiment Analysis
https://mc.ai/tensorflow-2%E2%80%8A-%E2%80%8Abert-movie-review-sentiment-analysis/
```
```
BERT===Bidirectional Encoder Representation from Transformers
雙向Transformer的Encoder
因為decoder是不能獲要預測的資訊的。
模型的主要創新點都在pre-train方法上，
用了Masked LM和Next Sentence Prediction兩種方法分別捕捉詞語和句子級別的representation
https://zhuanlan.zhihu.com/p/46652512
```
```
NLP 新時代的開始 - BERT
weifanhaha in 七天看懂自然語言處理的七年演進
Feb 29, 2020

https://www.coderbridge.com/@weifanhaha/e2e46480d185435680609a40997644be
```
```
BERT-analysis
https://github.com/aapp1420/BERT-analysis
```
# multilingual BERT
```
transformer-based BERT models for languages other than English
```

# GPT-1,GPT-2, GPT-3

### GPT-3
```
Language Models are Few-Shot Learners
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, 
Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, 
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, 
Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

https://arxiv.org/abs/2005.14165
https://github.com/openai/gpt-3
```
```
[research paper review] GPT-3 : Language Models are Few-Shot Learners
https://www.youtube.com/watch?v=Mq97CF02sRY

GPT-3: Language Models are Few-Shot Learners (Paper Explaine
https://www.youtube.com/watch?v=SY5PvZrJhLE
```
```
[DLHLP 2020] 來自獵人暗黑大陸的模型 GPT3
https://www.youtube.com/watch?v=DOG1L9lvsDY

http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/GPT3%20(v6).pdf
```

#
```
Video Generation by GAN
https://www.youtube.com/watch?v=TN8cJiomk_k
```
