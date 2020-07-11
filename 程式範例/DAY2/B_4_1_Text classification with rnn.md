#
```

```
### 
```
# 繪製線圖 (可將訓練時所傳回的損失值或準確率等歷史記錄繪製成線圖)
# history: 內含一或多筆要繪資料的字典, 例如：{'loss': [4,2,1,…], 'acc': [2,3,5,…]}
# keys: 以 tuple 或串列指定 history 中要繪製的 key 值, 例如：('loss', 'acc')
# title: 以字串指定圖表的標題文字
# xyLabel: 以 tuple 或串列指定 x, y 軸的說明文字, 例如：('epoch', 'Accuracy')
# ylim: 以 tuple 或串列指定 y 軸的最小值及最大值, 例如 (1, 3), 超出範圍的值會被忽略
# size: 以 tuple 指定圖的尺寸, 預設為 (6, 4) (即寬 6 高 4 英吋)
```
```
import matplotlib.pyplot as plt

def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType = ('-', '--', '.', ':')    # 線條的樣式, 畫多條線時會依序採用
    if len(ylim)==2: plt.ylim(*ylim)    # 設定 y 軸最小值及最大值
    if len(size)==2: plt.gcf().set_size_inches(*size)  # size預設為 (6,4)
    epochs = range(1, len(history_dict[keys[0]])+1)  # 計算有幾週期的資料
    for i in range(len(keys)):   # 走訪每一個 key (例如 'loss' 或 'acc' 等)
        plt.plot(epochs, history_dict[keys[i]], lineType[i])  # 畫出線條
    if title:   # 是否顯示標題欄
        plt.title(title)
    if len(xyLabel)==2:  # 是否顯示 x, y 軸的說明文字
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') #upper left')  # 顯示圖例 (會以 key 為每條線的說明)
    plt.show()  # 顯示出畫好的圖
```
```
#%% 載入 IMDB 資料集並進行序列對齊
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)  #←只處理常見的前 10000 個單字

maxlen = 500    #←設定序列長度為 500
train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, truncating='post')    #←只取前 500 個單字, 超過的截掉

#%% 建立模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.Flatten())  #←將嵌入層的輸出展平
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

#%% 訓練網路
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2)

#顯示訓練資料的損失和準確率曲線
plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

#顯示測試資料的損失和準確率
results = model.evaluate(test_data, test_labels, verbose=0)
print(results)
```
# 使用 SimpleRNN
```
rnn_model = Sequential()
rnn_model.add(layers.Embedding(10000, 16, input_length=maxlen))
rnn_model.add(layers.SimpleRNN(16))
rnn_model.add(layers.Dense(1, activation='sigmoid'))
rnn_model.summary()

rnn_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = rnn_model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2)

plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

results = rnn_model.evaluate(test_data, test_labels, verbose=0)
print(results)
```
# 使用 DROPOUT
```
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import util as u

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)

maxlen = 500
train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, truncating='post')

model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.SimpleRNN(16,
                           dropout=0.05,  #←設定丟棄率為 0.05
                           recurrent_dropout=0.05))  #←設定丟棄率為 0.05
model.add(layers.Dropout(0.1))  #←在密集層之前也加入一層丟棄層
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2)

plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

results = model.evaluate(test_data,test_labels, verbose=0)
print(results)
```

# 使用循環DROPOUT
```
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)

maxlen = 500
train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, truncating='post')

model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.SimpleRNN(16,
                           dropout=0.05,
                           recurrent_dropout=0.05,
                           return_sequences=True))  #←設定為 True 讓 RNN 能輸出每個時間點的資料
model.add(layers.SimpleRNN(16,
                           dropout=0.05,
                           recurrent_dropout=0.05
                           ))  #←最後一層 RNN, 返回一個時間點的資料即可
model.add(layers.Dropout(0.1))  #←在密集層之前也加入一層丟棄層
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2)

plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

results = model.evaluate(test_data,test_labels, verbose=0)
print(results)
```
# 使用LSTM
```
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import util4 as u

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)

maxlen = 500
train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, truncating='post')

model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.LSTM(16))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_split=0.2)

plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

results = model.evaluate(test_data,test_labels, verbose=0)
print(results)
```
# 使用GRU
```
model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.GRU(16))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
```
# 使用Bidirectional LSTM
```
model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.Bidirectional(layers.LSTM(16)))  #←在雙向層中傳入一個 LSTM 層
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
```
