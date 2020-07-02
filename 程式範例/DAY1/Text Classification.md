#
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
    plt.legend(keys, loc='best') # 顯示圖例 (會以 key 為每條線的說明)
    plt.show()  # 顯示出畫好的圖
```
#
```
from tensorflow.keras.datasets import imdb    #← 從 keras.datasets 套件中匯入 imdb
(a_train, b_train),(a_test, b_test)= imdb.load_data(num_words=10000) # 載入 IMDB

from tensorflow.keras.preprocessing.text import Tokenizer

tok = Tokenizer(num_words=10000)           #←指定字典的總字數
x_train = tok.sequences_to_matrix(a_train) #←將訓練樣本做 k-hot 編碼
x_test  = tok.sequences_to_matrix(a_test)  #←將測試樣本做 k-hot 編碼

y_train = b_train.astype('float32')   #←將訓練標籤轉為浮點向量
y_test  = b_test.astype('float32')    #←將測試標籤轉為浮點向量

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()                       #←建立模型物件
model.add(Dense(16, activation='relu', input_dim=10000))  #←輸入層
model.add(Dense(16, activation='relu'))    #←隱藏層
model.add(Dense(1, activation='sigmoid'))  #←輸出層

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=512,  #←每批次 512 筆樣本
                    epochs=10,       #←共訓練 10 週期
                    verbose = 2,     #←顯示精簡訊息 (無進度條)
                    validation_split=0.2)
                             #↑由訓練資料後面切出 20% 做為驗證用

plot(history.history,
       ('loss', 'val_loss'),          #←歷史資料中的 key
       'Training & Validation Loss',  #←線圖的標題
       ('Epoch','Loss'))              #←x,y 軸的名稱

plot(history.history,
       ('acc', 'val_acc'),            #←歷史資料中的 key
       'Training & Validation Acc',   #←線圖的標題
       ('Epoch','Acc'))               #←x,y 軸的名稱


```
