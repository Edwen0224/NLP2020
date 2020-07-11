# Topics
```
Text Classification
Text Preprocessing

Word embeddings

RNN Model

Transfer Learning in NLP
```
```
Text Geberation

Char-RNN
```
```
Image Caption
```
# Text Preprocessing
```

```
# Word embeddings(1)
```
one-hot 編碼
```
```
k-hot 編碼
Text Classification初探
```
# Word embeddings(2)
```
一文搞懂word embeddding和keras中的embedding
https://www.jianshu.com/p/b2c33d7e56a5
https://upload-images.jianshu.io/upload_images/4289471-c3f7c5682d3eed4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240

DeepNLP的表示学习·词嵌入来龙去脉·
https://blog.csdn.net/Scotfield_msn/article/details/69075227
```
```
Load text
https://tensorflow.google.cn/tutorials/load_data/text
```
```
Word embeddings
https://tensorflow.google.cn/tutorials/text/word_embeddings
```
```
Text classification with preprocessed text: Movie reviews
https://tensorflow.google.cn/tutorials/keras/text_classification
```
### 預先訓練自己的中文詞向量
```
tf.keras 技術者們必讀！深度學習攻略手冊
施威銘研究室 著 旗標科技  2020-02-13

第 6 章 預先訓練自己的中文詞向量
6-0 為什麼要預先訓練詞向量
6-1 Word2vec 實作原理
CBOW 連續詞袋模型
Skip-gram 跳字模型
進階 Skip-gram 模型
6-2 建立並訓練 Word2vec 神經網路
6-2-0 建立完整的 Word2vec 架構
6-2-1 取得原始資料 (語料)
維基百科語料
社區問答語料
6-2-2 資料預處理的介紹
解析 JSON 形式：json 套件
簡體轉繁體：opencc 套件
斷詞：Jieba 套件
6-2-3 資料預處理：產生訓練 Word2vec 所需的資料集
Wiki 資料集
```
# RNN Model
```
Recurrent Neural Networks (RNN) with Keras
https://tensorflow.google.cn/guide/keras/rnn
```
```
Masking is a way to tell sequence-processing layers that certain timesteps in an input are missing, and thus should be skipped when processing the data.

Padding is a special form of masking were the masked steps are at the start or at the beginning of a sequence. 
Padding comes from the need to encode sequence data into contiguous batches: in order to make all sequences in a batch fit a given standard length, 
it is necessary to pad or truncate some sequences.

https://tensorflow.google.cn/guide/keras/masking_and_padding
```
```
RNN.md
```
# Transfer Learning in NLP
```
Text classification with TensorFlow Hub: Movie reviews
https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
```
