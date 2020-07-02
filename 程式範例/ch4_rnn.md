# test code
```

```

# ch4
```
# %% 製作樣本資料和標籤資料
import numpy as np

sample = np.array([[[0],[1]],
                  [[1],[1]],
                  [[1],[2]]])

label = np.array([1, 2, 0])

# %% 將數字資料轉成 one-hot 編碼
from tensorflow.keras.utils import to_categorical

sample = to_categorical(sample)
print(sample)

label = to_categorical(label)

# %% 建立 RNN 模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.SimpleRNN(3,
                           input_shape=(2, 3),
                           activation='softmax'))
model.summary()

# %% 編譯 RNN 模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# %% 訓練 RNN 模型
history = model.fit(sample,
                    label,
                    epochs=500
                    )

# %% 輸出預測結果
predict = model.predict_classes(sample)
print(predict)

# %% 建立、編譯、訓練 RNN+DNN 模型
model = Sequential()
model.add(layers.SimpleRNN(10,
                           input_shape=(2, 3)))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(sample,
                    label,
                    epochs=100
                    )


```


#
```
#%% 製作每天晚餐的資料, 並轉為 one-hot 編碼
from tensorflow.keras.utils import to_categorical

dinner = [0,1,1,2,0,1,1,2]  #←以數字 0 代表咖哩飯;1 代表牛排;2 代表速食店
dinner = to_categorical(dinner)  #←將資料轉成 one-hot 編碼
print(dinner)

# %% 資料採樣
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

data_gen = TimeseriesGenerator(dinner, 
                               dinner,  #←目標資料來自於訓練資料
                               length=3, 
                               sampling_rate=1,
                               stride=1,
                               batch_size=2)

print(data_gen[0])  #←顯示第一批次的資料
print(len(data_gen))  #←顯示 Sequence 物件的長度
#%% 建立模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.SimpleRNN(10, input_shape=(None, 3)))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

#%% 編譯、訓練模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

train_history = model.fit(data_gen,
                          epochs=50
)

# %% 輸出預測結果
prediction = model.predict_classes(data_gen,batch_size=None)
print(prediction)


```


# IMDB 資料集(2):二元分類
```
import matplotlib.pyplot as plt

# 繪製線圖 (可將訓練時所傳回的損失值或準確率等歷史記錄繪製成線圖)
# history: 內含一或多筆要繪資料的字典, 例如：{'loss': [4,2,1,…], 'acc': [2,3,5,…]}
# keys: 以 tuple 或串列指定 history 中要繪製的 key 值, 例如：('loss', 'acc')
# title: 以字串指定圖表的標題文字
# xyLabel: 以 tuple 或串列指定 x, y 軸的說明文字, 例如：('epoch', 'Accuracy')
# ylim: 以 tuple 或串列指定 y 軸的最小值及最大值, 例如 (1, 3), 超出範圍的值會被忽略
# size: 以 tuple 指定圖的尺寸, 預設為 (6, 4) (即寬 6 高 4 英吋)
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

(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)  
#只處理常見的前 10000 個單字

maxlen = 500    #設定序列長度為 500
train_data = pad_sequences(train_data, maxlen=maxlen, truncating='post')
test_data = pad_sequences(test_data, maxlen=maxlen, truncating='post')    #只取前 500 個單字, 超過的截掉

#%% 建立模型
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential()
model.add(layers.Embedding(10000, 16, input_length=maxlen))
model.add(layers.Flatten())  #將嵌入層的輸出展平
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

#%%  顯示訓練資料的損失和準確率曲線

plot(history.history,
       ['loss','val_loss'],
       title='Training & Validation Loss',
       xyLabel=['Epoch','Loss'])

plot(history.history,
       ['acc','val_acc'],
       title='Training & Validation Acc',
       xyLabel=['Epoch','Acc'])

#%%  顯示測試資料的損失和準確率
results = model.evaluate(test_data, test_labels, verbose=0)
print(results)

#%%  建立並訓練 RNN 模型
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


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```


#
```



```
