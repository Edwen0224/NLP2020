#
```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))
```

#
```
from tensorflow.keras import Model                # Model 為函數式 API 的模型類別
from tensorflow.keras.layers import Input, Dense  # 匯入 Input 及 Dense 層類別

A = Input(shape=(784,))                # 將 Input 層的輸出張量 (傳回值) 指定給 A
B = Dense(512, activation='relu')(A)   # 將 A 傳入第一 Dense 層做為輸入張量, 輸出張量指定給 B
C = Dense(10, activation='softmax')(B) # 將 B 傳入第二 Dense 層做為輸入張量, 輸出張量指定給 C

model = Model(A, C)   # 用【最初的輸入張量】和【最後的輸出張量】來建立模型
```

#
```
from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)
txt = LSTM(32)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層
out = Dense(1)(out)                         # 輸出層, 可輸出一個預測的銷量值

model = Model([att_in, txt_in, img_in], out)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 解迴歸問題
wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用

model.summary()

#用程式產生訓繙資料
import numpy as np

rng = np.random.RandomState(45) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數

    # base的範圍  var的範圍  輸入樣本的 shape
    #         ↓     ↓      ↓
def toy_data(base, var, shape):
    arr = rng.randint(var, size=shape)        # 依 shape 產生 0~(var-1) 的隨機值
    for i in range(shape[0]):                 # 走訪每一個樣本
        arr[i] = arr[i] + rng.randint(base+1) # 將樣本中的每個特徵都加上一個固定的隨機值 (0~base)
    return arr

total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

# 顯示各資料的 shape 及最小、最大值
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))

x_att = x_att.astype('float32') / 100   # 將資料正規化
x_img = x_img.astype('float32') / 255   # 將資料正規化

                  #  以串列依序傳入 3 種樣本資料     分出 20% 做為驗證資料
                  #  -------------------              ↓
history = model.fit([x_att, x_txt, x_img], y, validation_split=0.2,
                    batch_size=128, epochs=1000, verbose=2)

# 訓練方法 2：使用 dict 方式送入資料進行訓練, 鍵為 Input 層的名稱, 值為 Numpy 資料
#model.fit({'att': x_att, 'txt': x_txt, 'img': x_img}, y,
#          validation_split=0.2, batch_size=128, epochs=1000, verbose=2)

import util5 as u   # 匯入自訂模組 (參見 2-0 節的最後單元)

u.plot(history.history, ('mae', 'val_mae'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Mae', ('Epoch','mae'),
       ylim=(0, 300), size=(12, 4))


his = history.history['val_mae']
idx = np.argmin(his)    # 找出最低 val_mae 值的索引
print('最低 val_mae 為第', idx+1, '週期的', his[idx])

    #要轉換的歷史記錄  EMA 係數
    #         ↓      ↓
def to_EMA(points, a=0.3):
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

his_EMA = to_EMA(history.history['val_mae'])  # 將 val_mae 的值轉成 EMA 值
idx = np.argmin(his_EMA)                      # 找出最低 EMA 值的索引
print('最小 EMA 為第', idx+1, '週期的', his_EMA[idx])


history.history['EMA400'] = his_EMA[400:801]   #取 400~800 的 EMA 資料
history.history['mae400'] = his[400:801]       #取 400~800 的 val_mae 資料
u.plot(history.history, ('EMA400', 'mae400'),  #繪製 EMA 和 val_mae 的比較圖
       'Validation EMA & mae', ('Epoch 400~800','EMA & mae'),
       ylim=(0, 30), size=(12, 4))
             # ↑ y 軸限定只畫出 0~30 的範圍

#####################################################

print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
history = model.fit([x_att, x_txt, x_img], y,     # 用所有的訓練資料重新訓練到第 {idx+1} 週期
          batch_size=128, epochs=idx+1, verbose=2)

print('重新產生 10000 筆測試資料來評估成效')
rng = np.random.RandomState(67) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數
total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

x_att = x_att.astype('float32') / 100   # 將資料正規化
x_img = x_img.astype('float32') / 255   # 將資料正規化

loss, mae = model.evaluate([x_att, x_txt, x_img], y)
print(f'用 10000 筆測試資料評估的結果：mae={round(mae, 3)}')

pred = model.predict([x_att[:3], x_txt[:3], x_img[:3]])
#pred = model.predict({'att':x_att[:3], 'txt':x_txt[:3], 'img':x_img[:3]})
print('預測銷量:', pred.round(1))
print('實際銷量:', y[:3].round(1))
```

#
```
from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)
txt = LSTM(32)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([txt, img], axis=-1)      # 用輔助函式串接張量
out = Dense(28, activation='relu')(out)     # 密集層
out = Dense(1)(out)                         # 輸出層, 可輸出一個預測的銷量值

model = Model([txt_in, img_in], out)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 解迴歸問題
wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用

model.summary()

#用程式產生訓繙資料
import numpy as np

rng = np.random.RandomState(45) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數

    # base的範圍  var的範圍  輸入樣本的 shape
    #         ↓     ↓      ↓
def toy_data(base, var, shape):
    arr = rng.randint(var, size=shape)        # 依 shape 產生 0~(var-1) 的隨機值
    for i in range(shape[0]):                 # 走訪每一個樣本
        arr[i] = arr[i] + rng.randint(base+1) # 將樣本中的每個特徵都加上一個固定的隨機值 (0~base)
    return arr

total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

# 顯示各資料的 shape 及最小、最大值
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))

x_att = x_att.astype('float32') / 100   # 將資料正規化
x_img = x_img.astype('float32') / 255   # 將資料正規化

history = model.fit([x_txt, x_img], y, validation_split=0.2,
                    batch_size=128, epochs=500, verbose=2)

import util5 as u   # 匯入自訂模組 (參見 2-1 節的最後單元)

u.plot(history.history, ('mae', 'val_mae'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Mae', ('Epoch','mae'),
       ylim=(0, 300), size=(12, 4))


his = history.history['val_mae']
idx = np.argmin(his)    # 找出最低 val_mae 值的索引
print('最低 val_mae 為第', idx+1, '週期的', his[idx])


```

#
```
from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)s
txt = LSTM(28)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層

sell_out = Dense(1, name='sell')(out)     # 迴歸分析的銷量輸出層：輸出預測的銷量
eval_out = Dense(3, activation='softmax', name='eval')(out) # 多元分類的評價輸出層：輸出好評、中評、或負評

model = Model([att_in, txt_in, img_in], [sell_out, eval_out]) # 2 個輸出層

#用程式產生訓繙資料
import numpy as np

rng = np.random.RandomState(45) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數

    # base的範圍  var的範圍  輸入樣本的 shape
    #         ↓     ↓      ↓
def toy_data(base, var, shape):
    arr = rng.randint(var, size=shape)        # 依 shape 產生 0~(var-1) 的隨機值
    for i in range(shape[0]):                 # 走訪每一個樣本
        arr[i] = arr[i] + rng.randint(base+1) # 將樣本中的每個特徵都加上一個固定的隨機值 (0~base)
    return arr

total = 10000   # 產生 total 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

y2 = np.ones(total) # 建立評價陣列, 元素值預設為 1 (中評)
att1 = x_att[:, 1]  # 由商品屬性中取出性價比資料
y2[att1>80] = 2   # 性價比大於 80 設為 2 (好評)
y2[att1<20] = 0   # 性價比小於 20 設為 0 (負評)
print('評價 y2 的好評數：', np.sum(y2==2), ', 中評數：', np.sum(y2==1), ', 負評數：', np.sum(y2==0))

print('原來的銷量 y:', y.shape,  ', min =', np.min(y),     ', max =', np.max(y))
y[y2==2] *= 1.5
y[y2==0] *= 0.5
print('調整的銷量 y:', y.shape,  ', min =', np.min(y),     ', max =', np.max(y))

print('total:', total)
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))
print('y2:  :', y2.shape,    ', min =', np.min(y2),    ', max =', np.max(y2))

#建立自訂評量函式
from tensorflow.keras import metrics

def macc(y_true, y_pred):
  return (4000 - metrics.mae(y_true, y_pred)) / 4000

model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy'],
              metrics=[['mae', macc], ['acc']],
              loss_weights=[0.01, 100])

#model.compile(optimizer='rmsprop',
#              loss={'sell': 'mse', 'eval': 'categorical_crossentropy'},
#              metrics={'sell': ['mae', macc], 'eval': 'acc'},
#              loss_weights={'sell':0.01, 'eval':100})

wt = model.get_weights()    #←儲存模型的初始權重, 以供稍後重新訓練模型時使用
model.summary()


x_att = x_att.astype('float32') / 100   # 將屬性資料正規化
x_img = x_img.astype('float32') / 255   # 將圖片資料正規化

from tensorflow.keras.utils import to_categorical
y2 = to_categorical(y2)   # 做 one-hot 編碼

                  #  以串列依序傳入 3 種樣本資料     分出 20% 做為驗證資料
                  #  -------------------              ↓
history = model.fit([x_att, x_txt, x_img], [y, y2], validation_split=0.2,
                    batch_size=128, epochs=800, verbose=2)


## 訓練方法 2：使用 dict 方式送入資料進行訓練, 鍵為 Input 層的名稱, 值為 Numpy 資料
#history = model.fit({'att': x_att, 'txt': x_txt, 'img': x_img}, {'sell': y, 'eval': y2},
#                    validation_split=0.2, batch_size=128, epochs=800, verbose=2)

import util5 as u   # 匯入自訂模組 (參見 2-0 節的最後單元)

u.plot(history.history, ('sell_macc', 'val_sell_macc', 'eval_acc', 'val_eval_acc'),   #←繪製訓練及驗證的 mae 歷史線圖
       'Training & Validation Acc', ('Epoch','acc'),
       ylim=(0.6, 1.0), size=(12, 4))

#######################################################
                             # zip() 每次會從各參數中取出一個元素來傳回
his = history.history        # ↓
his_avg = [(a+b)/2 for a, b in zip(his['val_sell_macc'], his['val_eval_acc'])] # 計算 2 種準確率的平均值

idx = np.argmax(his_avg)     # 找出平均準確率最高的索引
print('最高準確率為第', idx+1, '週期的', his_avg[idx],
      ' (銷量：', his['val_sell_macc'][idx], ', 評價：', his['val_eval_acc'][idx], ')')

def to_EMA(points, a=0.3):  # 有關 EMA 的說明請參見前一單元
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

his_EMA = to_EMA(his_avg)  # 將 his_avg 值轉成 EMA 值
idx = np.argmax(his_EMA)   # 找出最高 EMA 值的索引
print('最高 EMA 為第', idx+1, '週期的', his_EMA[idx],
      ' (銷量：', his['val_sell_macc'][idx],
      ', 評價：', his['val_eval_acc'][idx], ')')

# #####################################################

print(f'用所有的訓練資料重新訓練到第 {idx+1} 週期')
model.set_weights(wt)  #←還原初始權重 (效果等於重建模型, 以便重新訓練)
history = model.fit([x_att, x_txt, x_img], [y, y2],     # 用所有的訓練資料重新訓練到第 {idx+1} 週期
          batch_size=128, epochs=idx+1, verbose=2)

rng = np.random.RandomState(67) # 以固定種子產生隨機物件, 以便每次執行都能產生相同的亂數
total = 10000   # 產生 10000 個樣本
x_att = toy_data(10,   90, shape=(total, 2))
x_txt = toy_data(900, 100, shape=(total, 100))
x_img = toy_data(200,  56, shape=(total, 32, 32, 3))

y = (np.mean(x_att, axis=-1)*10 +    # 依樣本算出標籤 (銷量) 資料
     np.mean(x_txt, axis=-1) +
     np.mean(x_img, axis=(-1,-2,-3))*4)

y2 = np.ones(total)
att0 = x_att[:, 0]
att1 = x_att[:, 1]

y2[att1>80] = 2
y2[att1<20] = 0
print('好評數：', np.sum(y2==2), ', 中評數：', np.sum(y2==1), ', 負評數：', np.sum(y2==0))

y[y2==2] *= 1.5
y[y2==0] *= 0.5

# 顯示各資料的 shape 及最小、最大值
print('x_att:', x_att.shape, ', min =', np.min(x_att), ', max =', np.max(x_att))
print('x_txt:', x_txt.shape, ', min =', np.min(x_txt), ', max =', np.max(x_txt))
print('x_img:', x_img.shape, ', min =', np.min(x_img), ', max =', np.max(x_img))
print('y:   :', y.shape,     ', min =', np.min(y),     ', max =', np.max(y))
print('y2:  :', y2.shape,    ', min =', np.min(y2),    ', max =', np.max(y2))

x_att = x_att.astype('float32') / 100
x_img = x_img.astype('float32') / 255
y2 = to_categorical(y2)

res = model.evaluate([x_att, x_txt, x_img], [y, y2], verbose=2)
print(f'用 {total} 筆測試資料評估的結果：{res}')

pred = model.predict([x_att[:3], x_txt[:3], x_img[:3]])
#pred = model.predict({'att':x_att[:3], 'txt':x_txt[:3], 'img':x_img[:3]})

print('預測銷量:', pred[0].round(1))
print('實際銷量:', y[:3].round(1))

print('預測評價:', pred[1].round(1))
print('實際評價:', y2[:3])
```

#
```
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate

inp = Input(shape=(256, 256, 3))  # 輸入層 (256x256 的 3 通道圖片)

b1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) # 第 1 分支
b1 = Conv2D(64, (3, 3), padding='same', activation='relu')(b1)

b2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) # 第 2 分支
b2 = Conv2D(64, (5, 5), padding='same', activation='relu')(b2)

b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp)  # 第 3 分支
b3 = Conv2D(64, (1, 1), padding='same', activation='relu')(b3)

out = concatenate([b1, b2, b3], axis=1)  # 將 3 個分支串接起來


```

#
```
from tensorflow.keras.layers import Conv2D, Input, add

inp = Input(shape=(256, 256, 3))

b = Conv2D(3, (3, 3), padding='same')(inp)
b = Conv2D(3, (3, 3), padding='same')(b)

out = add([b, inp])  # 將分支 b 和 inp 相加
```

#
```
from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate

                     #↓不定長度的句子,以 256 字的字典編碼
a_in = Input(shape=(None, 256))  # 分支 a 的輸入
b_in = Input(shape=(None, 256))  # 分支 b 的輸入

shared_lstm = LSTM(64)  # 建立共用的 LSTM 層

a = shared_lstm(a_in)   # 建立分支 a, 輸出張量 shape 為 (批次量, 64)
b = shared_lstm(b_in)   # 建立分支 a, 輸出張量 shape 為 (批次量, 64)

c = concatenate([a, b]) # 將二分支的輸出串接起來
out = Dense(1, activation='sigmoid')(c)  # 進行 2 元分類 (是否意義相同？)

model = Model(inputs=[a_in, b_in], outputs=out)  # 建立模型

model.compile(optimizer='rmsprop', loss='binary_crossentropy',  # 編譯模型
              metrics=['acc'])

model.summary()
```

#
```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda

A = Input((4,))
B = Input((5,))

lmb = Lambda(lambda x: [x/2, x*2]) # Lambda 層
C,D = lmb(A)  # 輸入 A, 輸出 C、D
E,F = lmb(B)  # 輸入 B, 輸出 E、F

C = Dense(6)(C)   #}
D = Dense(7)(D)   #} 分別將 C,D,E,F 都
E = Dense(8)(E)   #} 連到一個 Dense 層
F = Dense(9)(F)   #}

model = Model([A, B], [C, D, E, F]) # 建立模型
model.summary()
```

#
```
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate

a = Input(shape=(784,))
b = Input(shape=(784,))

shr  = Dense(512, activation='relu')    # 建立解析圖片用的共用層
out = concatenate([shr(a), shr(b)])     # 將 a,b 輸入到共用層, 再將其輸出串接起來

out = Dense(10, activation='relu')(out)    # 建立學習分類用的 Dense 層
out = Dense(1, activation='sigmoid')(out)  # 進行 2 元分類 (是否同數字)

model = Model([a, b], out)               # 建立模型
model.compile(optimizer='rmsprop',       # 編譯模型
              loss='binary_crossentropy', metrics=['acc'])

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 載入MNIST資料集
xa = train_images.reshape((60000, 28 * 28)).astype('float32') / 255  # 預處理圖片樣本
ya = train_labels        # 標籤不預處理, 因為稍後要用來比對是否為相同數字

idx = np.arange(40000)                      #}將 xa、ya 的前 4 萬筆隨機重排後,
np.random.shuffle(idx)                      #}連同後 2 萬筆一起另存到 xb、yb,
xb = np.concatenate((xa[idx], xa[40000:]))  #}這樣至少會有 2 萬筆以上的標籤為相同數字
yb = np.concatenate((ya[idx], ya[40000:]))  #}

y = np.where(ya==yb, 1.0, 0.0)   # 建立標籤：1 為是(相同數字), 0 為否

idx = np.arange(60000)               #} 再次將 xa/ya、xb/yb 同步隨機重排
np.random.shuffle(idx)               #}
xa, xb, y = xa[idx], xb[idx], y[idx] #}

print(f'訓練資料共 {len(y)} 筆, 其中有 {int(y.sum())} 筆為相同數字')

his = model.fit([xa, xb], y, validation_split=0.1,     #} 取 10% 做驗證, 訓練 20 週期
                epochs=20, batch_size=128, verbose=2)  #}

import util5 as u
u.plot(his.history, ('acc', 'val_acc'),     # 繪製訓練和驗證的準確率線圖
       'Training & Validating Acc', ('Epoch','Acc'))



```

#
```
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten

img_model = Sequential()                             #}
img_model.add(Conv2D(32, (3, 3), activation='relu',  #} 建立序列式模型
              input_shape=(28, 28, 1)))              #} 輸入 shape 為 (28, 28, 1)
img_model.add(Flatten())                             #} 輸出 shape 為 (10,)
img_model.add(Dense(10, activation='softmax'))       #}

#img_model.fit(.....)  # 進行訓練

inp = Input(shape=(28, 28, 1))  # 建立輸入層, 輸入 shape 為 (28, 28, 1)
out = img_model(inp)          # 加入已訓練好的模型 (包含模型中的權重)
out = Dense(10)(out)            # 再連接到 Dense 層
model = Model(inp, out)         # 建立函數式模型
```

#
```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') /255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')  /255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()                      # 建立 CNN 模型
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',         # 編譯模型
              optimizer='Adam', metrics=['acc'])

model.fit(x_train, y_train, batch_size=128, epochs=12) # 訓練模型

score = model.evaluate(x_test, y_test, verbose=0)      # 評估成效
print('對測試資料集的準確率：', score[1])

model.save('模型_MNIST_CNN.h5')     # 將模型存檔, 以供稍後使用

```

#
```

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate

a = Input(shape=(28, 28, 1))  #} 建立 2 個輸入層, 輸入 shape 均為 (28, 28, 1)
b = Input(shape=(28, 28, 1))  #}

cnn = load_model('模型_MNIST_CNN.h5')    # 載入已訓練好的 CNN 模型
cnn.trainable = False                   # 將模型設為不可訓練 (鎖住權重)

out = concatenate([cnn(a), cnn(b)])  # 將 a,b 輸入到 CNN 模型層, 並將輸出串接起來
out = Dense(128, activation='relu')(out)    # 建立學習分類用的 Dense 層
out = Dense(1, activation='sigmoid')(out)   # 進行 2 元分類 (是否同數字)

model = Model([a, b], out)               # 建立模型
model.compile(optimizer='rmsprop',       # 編譯模型
              loss='binary_crossentropy', metrics=['acc'])

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # 載入MNIST資料集
xa = train_images.reshape((60000,784)).astype('float32') / 255  # 預處理圖片樣本
ya = train_labels        # 標籤不預處理, 因為稍後要用來比對是否為相同數字

idx = np.arange(40000)                      #}將 xa、ya 的前 4 萬筆隨機重排後,
np.random.shuffle(idx)                      #}連同後 2 萬筆一起另存到 xb、yb,
xb = np.concatenate((xa[idx], xa[40000:]))  #}這樣至少會有 2 萬筆以上的標籤為相同數字
yb = np.concatenate((ya[idx], ya[40000:]))  #}

y = np.where(ya==yb, 1.0, 0.0)   # 建立標籤：1 為是(相同數字), 0 為否

idx = np.arange(60000)               #} 再次將 xa/ya、xb/yb 同步隨機重排
np.random.shuffle(idx)               #}
xa, xb, y = xa[idx], xb[idx], y[idx] #}

xa = xa.reshape((60000,28,28,1))  #} 將樣本改為符合 CNN 輸入的 shape
xb = xb.reshape((60000,28,28,1))  #}
n = 2000                             # 設定只取前 2000 筆來做訓練
print(f'訓練資料共 {len(y[:n])} 筆, 其中有 {int(y[:n].sum())} 筆為相同數字')
                #         ↑                       ↑
                # 只取前 n 筆做訓練
                #   ↓       ↓     ↓
his = model.fit([xa[:n], xb[:n]], y[:n], validation_split=0.1,   #} 取 10% 做驗證, 訓練 20 週期
                epochs=20, batch_size=128, verbose=2)  #}

import util5 as u
u.plot(his.history, ('acc', 'val_acc'),     # 繪製訓練和驗證的準確率線圖
       'Training & Validating Acc', ('Epoch','Acc'))

                  # 將剩下的資料拿來評估成效
                  #       ↓                       ↓
print(f'測試資料共 {len(y[n:])} 筆, 其中有 {int(y[n:].sum())} 筆為相同數字')
score = model.evaluate([xa[n:], xb[n:]], y[n:], verbose=0)      # 評估成效
print('對測試資料集的準確率：', score[1])
```

#
```
from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)s
txt = LSTM(28)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層

sell_out = Dense(1, name='sell')(out)     # 迴歸分析的銷量輸出層：輸出預測的銷量
eval_out = Dense(3, activation='softmax', name='eval')(out) # 多元分類的評價輸出層：輸出好評、中評、或負評

model = Model([att_in, txt_in, img_in], [sell_out, eval_out]) # 2 個輸出層

#####↓↓繪製模型圖↓↓#####

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_1.png')  #←繪製模型結構圖

plot_model(model, to_file='model_2.png',  #←繪製包含 shape 但沒有神經層名稱的模型結構圖
           show_shapes=True,
           show_layer_names=False)


```

# 5-4
```
from tensorflow.keras.models import Model   # 匯入 Model 類別
from tensorflow.keras.layers import (Input, Dense, Embedding,
                LSTM, Conv2D, MaxPooling2D, Flatten, concatenate)

att_in = Input(shape=(2,), name='att')      # 商品屬性的輸入 shape=(None, 2)
att = Dense(16, activation='relu')(att_in)  # 密集層的輸出 shape=(None, 16)

txt_in = Input(shape=(100,), name='txt')    # 文案的輸入 shape=(None, 100)
txt = Embedding(1000, 32)(txt_in)           # 嵌入層 (字典只取 1000 字)s
txt = LSTM(28)(txt)                         # LSTM 層的輸出 shape=(None, 32)

img_in = Input(shape=(32, 32, 3), name='img')       # 圖片的輸入 shape=(None, 32,32,3)
img = Conv2D(32, (3, 3), activation='relu')(img_in) # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Conv2D(32, (3, 3), activation='relu')(img)    # 卷積層
img = MaxPooling2D((2,2))(img)                      # 池化層
img = Flatten()(img)                                # 展平層的輸出 shape=(None, 1152)

out = concatenate([att, txt, img], axis=-1) # 用輔助函式串接 3 個張量
out = Dense(28, activation='relu')(out)     # 密集層

sell_out = Dense(1, name='sell')(out)     # 迴歸分析的銷量輸出層：輸出預測的銷量
eval_out = Dense(3, activation='softmax', name='eval')(out) # 多元分類的評價輸出層：輸出好評、中評、或負評

model = Model([att_in, txt_in, img_in], [sell_out, eval_out]) # 2 個輸出層

#####↓↓繪製模型圖↓↓#####

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_1.png')  #←繪製模型結構圖

plot_model(model, to_file='model_2.png',  #←繪製包含 shape 但沒有神經層名稱的模型結構圖
           show_shapes=True,
           show_layer_names=False)


```

#
```
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense

dnn = Sequential()   #← 建立序列模型
dnn.add(Dense(32, activation='relu', input_dim=48))
dnn.add(Dense(16, activation='relu'))

inp = Input(shape=48)
out = dnn(inp)    #←將序列模型加到目前模型中, 形成巢狀的神經層結構
out = Dense(10)(out)
model = Model(inp, out) # 建立包含巢狀神經層的函數式模型

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_3.png',
           rankdir='LR')  #←由左而右繪製

plot_model(model, to_file='model_4.png',
           rankdir='LR',         #←由左而右繪製
           expand_nested=True)   #←要繪製出巢狀(套疊)神經層的內部結構

```

# 5-5
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


