#
```
資料來源:Load text
https://www.tensorflow.org/tutorials/load_data/text

```
# 使用 tf.data 載入文本資料
```
本教程提供了一個如何使用 tf.data.TextLineDataset`來載入文字檔的示例。

TextLineDataset 通常被用來以文字檔構建資料集（原文件中的一行為一個樣本) 。
這適用於大多數的基於行的文本資料（例如，詩歌或錯誤日誌) 。

下面我們將使用相同作品（荷馬的伊利亞特）三個不同版本的英文翻譯，
然後訓練一個模型來通過單行文本確定譯者。

三個版本的翻譯分別來自于:
 - [William Cowper](https://en.wikipedia.org/wiki/William_Cowper) — [text](https://storage.googleapis.com/download.tensorflow.org/data/illiad/cowper.txt)
 - [Edward, Earl of Derby](https://en.wikipedia.org/wiki/Edward_Smith-Stanley,_14th_Earl_of_Derby) — [text](https://storage.googleapis.com/download.tensorflow.org/data/illiad/derby.txt)
- [Samuel Butler](https://en.wikipedia.org/wiki/Samuel_Butler_%28novelist%29) — [text](https://storage.googleapis.com/download.tensorflow.org/data/illiad/butler.txt)

本教程中使用的文字檔已經進行過一些典型的預處理，
主要包括刪除了文檔頁眉和頁腳，行號，章節標題。
請下載這些已經被局部改動過的檔。
```


# 環境設定與套件載入
```
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorflow_datasets as tfds
import os

```
# 下載資料
```
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)
  
parent_dir = os.path.dirname(text_dir)

parent_dir
```
```
"""
## 將文本載入到資料集中
反覆運算整個文件，將整個文件載入到自己的資料集中。
每個樣本都需要單獨標記，所以請使用 `tf.data.Dataset.map` 來為每個樣本設定標籤。
這將反覆運算資料集中的每一個樣本並且返回（ `example, label` ）對。
"""

def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

"""將這些標記的資料集合並到一個資料集中，然後對其進行隨機化操作。"""

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

"""使用 `tf.data.Dataset.take` 與 `print` 來查看 `(example, label)` 對的外觀。`numpy` 屬性顯示每個 Tensor 的值。"""

for ex in all_labeled_data.take(5):
  print(ex)
```
# 
```
"""
## 將文本編碼成數位
機器學習基於的是數位而非文本，所以字串需要被轉化成數位清單。
為了達到此目的，我們需要構建文本與整數的一一映射。

### 建立詞彙表
首先，通過將文本標記為單獨的單詞集合來構建詞彙表。
在 TensorFlow 和 Python 中均有很多方法來達成這一目的。

在本教程中:
1. 反覆運算每個樣本的 `numpy` 值。
2. 使用 `tfds.features.text.Tokenizer` 來將其分割成 `token`。
3. 將這些 `token` 放入一個 Python 集合中，借此來清除重複項。
4. 獲取該詞彙表的大小以便於以後使用。
"""
```
```

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size
```
## 
```
"""
### 樣本編碼
通過傳遞 `vocabulary_set` 到 `tfds.features.text.TokenTextEncoder` 來構建一個編碼器。
編碼器的 `encode` 方法傳入一行文本，返回一個整數清單。
"""

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

"""你可以嘗試運行這一行代碼並查看輸出的樣式。"""

example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)

encoded_example = encoder.encode(example_text)
print(encoded_example)

"""現在，在資料集上運行編碼器（通過將編碼器打包到 `tf.py_function` 並且傳參至資料集的 `map` 方法的方式來運行）。"""

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

"""
## 將資料集分割為測試集和訓練集且進行分支

使用 `tf.data.Dataset.take` 和 `tf.data.Dataset.skip` 來建立一個小一些的測試資料集和稍大一些的訓練資料集。

在資料集被傳入模型之前，資料集需要被分批。
最典型的是，每個分支中的樣本大小與格式需要一致。
但是資料集中樣本並不全是相同大小的（每行文本字數並不相同）。
因此，使用 `tf.data.Dataset.padded_batch`（而不是 `batch` ）將樣本填充到相同的大小。
"""

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

"""現在，test_data 和 train_data 不是（ `example, label` ）對的集合，而是批次的集合。每個批次都是一對（*多樣本*, *多標籤* ），表示為陣列。"""

sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]

"""由於我們引入了一個新的 token 來編碼（填充零），因此詞彙表大小增加了一個。"""

vocab_size += 1

"""
## 建立模型

第一層將整數表示轉換為密集向量嵌入。更多內容請查閱 [Word Embeddings](../../tutorials/sequences/word_embeddings) 教程。

下一層是 [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 層，它允許模型利用上下文中理解單詞含義。 
LSTM 上的雙向包裝器有助於模型理解當前資料點與其之前和之後的資料點的關係。

最後，我們將獲得一個或多個緊密連接的層，其中最後一層是輸出層。
輸出層輸出樣本屬於各個標籤的概率，最後具有最高概率的分類標籤即為最終預測結果。
"""

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, 64))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

""""""

# 一個或多個緊密連接的層
# 編輯 `for` 行的列表去檢測層的大小
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# 輸出層。第一個參數是標籤個數。
model.add(tf.keras.layers.Dense(3, activation='softmax'))

"""
編譯模型。
對於一個 softmax 分類模型來說，通常使用 `sparse_categorical_crossentropy` 作為其損失函數。
你可以嘗試其他的優化器，但是 `adam` 是最常用的。
"""

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
## 訓練模型

利用提供的資料訓練出的模型有著不錯的精度（大約 83% ）。
"""

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))

```
