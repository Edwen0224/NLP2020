#
```
TensorFlow 2 / 2.0 中文文檔
https://geektutu.com/post/tf2doc.html

https://geektutu.com/post/tf2doc-ml-basic-text.html
```
### tf.keras.layers.Add()
```
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add

https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
```
```
from tensorflow.keras.layers import Conv2D, Dense, Input, add , concatenate

input1 = tf.keras.layers.Input(shape=(16,))
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)

input2 = tf.keras.layers.Input(shape=(32,))
x2 = tf.keras.layers.Dense(8, activation='relu')(input2)

# equivalent to `added = tf.keras.layers.add([x1, x2])`
added = tf.keras.layers.Add()([x1, x2])

out = tf.keras.layers.Dense(4)(added)
model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
model.summary()
tf.keras.utils.plot_model(model,to_file='model.png')
```
### tf.keras.layers.concatenate
```
https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate
```
```
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Input, add , concatenate

x = np.arange(20).reshape(2, 2, 5)
y = np.arange(20, 30).reshape(2, 1, 5)

tf.keras.layers.concatenate([x, y],  axis=1)
```


#
```
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate

inp = Input(shape=(256, 256, 3))  # 輸入層 (256x256 的 3 通道圖片)

b1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp)
b1 = Conv2D(64, (3, 3), padding='same', activation='relu')(b1)

b2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) 
b2 = Conv2D(64, (5, 5), padding='same', activation='relu')(b2)

b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp) 
b3 = Conv2D(64, (1, 1), padding='same', activation='relu')(b3)

out = concatenate([b1, b2, b3], axis=1)  # 將 3 個分支串接起來
```


#
```
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate

inp = Input(shape=(256, 256, 3))  # 輸入層 (256x256 的 3 通道圖片)

b1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp)
b1 = Conv2D(64, (3, 3), padding='same', activation='relu')(b1)

b2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inp) 
b2 = Conv2D(64, (5, 5), padding='same', activation='relu')(b2)

b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp) 
b3 = Conv2D(64, (1, 1), padding='same', activation='relu')(b3)

out = concatenate([b1, b2, b3], axis=1)  # 將 3 個分支串接起來
```


#
```

```

```

```


#
```

```

```

```


#
```

```

```

```
