#
```
Text classification with preprocessed text: Movie reviews
https://tensorflow.google.cn/tutorials/keras/text_classification
```

```
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)


imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 參數 num_words=10000 保留了訓練資料中最常出現的 10,000 個單詞。
# 為了保持資料規模的可管理性，低頻詞將被丟棄。
‵```
# 理解與認識使用的資料集
```
# 查看幾筆訓練資料
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 查看第一筆訓練資料
print(train_data[0])

# 評論文本被轉換為整數值，其中每個整數代表詞典中的一個單詞。


# 顯示第一條和第二條評論的中單詞數量

len(train_data[0]), len(train_data[1])

# 電影評論可能具有不同的長度。
# 但由於神經網路的輸入必須是統一的長度，使用 pad_sequences 函數來使長度標準化
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
# 看看是否都統一長度了?
len(train_data[0]), len(train_data[1])

# 看第一條資料
print(train_data[0])
```

# 構建模型
```
# 輸入形狀是用於電影評論的詞彙數目（10,000 詞）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```
```
tf.keras.layers.GlobalAveragePooling1D
Global average pooling operation for temporal data.
https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
```
# 配置模型
```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

# 創建一個驗證集
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
# 評估模型
```
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)
```
# 創建一個準確率（accuracy）和損失值（loss）隨時間變化的圖表
```
history_dict = history.history
history_dict.keys()
```
```
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "藍點"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“藍色實線”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```
```
plt.clf()   # 清除數字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```
