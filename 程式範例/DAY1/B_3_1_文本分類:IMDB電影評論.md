#
```
https://www.tensorflow.org/tutorials/keras/text_classification
中文版

https://www.jishuwen.com/d/2O0H/zh-tw
```

```
資料集:網路電影資料庫(Internet Movie Database)
包含 50,000 條影評文本。
從該資料集切割出的25,000條評論用作訓練，
另外 25,000 條用作測試。
訓練集與測試集是*平衡的（balanced）*，意味著它們包含相等數量的積極和消極評論。

此範例評論文本
將影評分為*積極（positive）*或*消極（nagetive）*兩類。
這是一個二元（binary）分類問題，一種重要且應用廣泛的機器學習問題。

此範例使用 [tf.keras]，它是一個 Tensorflow 中用於構建和訓練模型的高級API。
```

#
```
import tensorflow as tf
from tensorflow import keras

import numpy as np
print(tf.__version__)


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0])

len(train_data[0]), len(train_data[1])


# 一個映射單詞到整數索引的詞典
word_index = imdb.get_word_index()

# 保留第一個索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

```
```
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
```
```
len(train_data[0]), len(train_data[1])

print(train_data[0])
```
# 構建模型
```
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```
# 配置模型
```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
# 建立驗證集
```
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
```
# 訓練模型
```
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
```
# 建立 準確率（accuracy）和損失值（loss）隨時間變化的圖表
```
history_dict = history.history
history_dict.keys()


import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 清除數字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```


