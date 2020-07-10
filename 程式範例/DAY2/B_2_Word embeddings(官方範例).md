#
```
Word embeddings
https://www.tensorflow.org/tutorials/text/word_embeddings
```

```
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()
```
# Embedding layer的使用
```
embedding_layer = layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1,2,3]))
result.numpy()

result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
result.shape
```
```
tf.keras.layers.Embedding
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

Turns positive integers (indexes) into dense vectors of fixed size.
將正整數（索引值）轉換為固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
該層只能用作模型中的第一層。This layer can only be used as the first layer in a model.


tf.keras.layers.Embedding(
    input_dim,     int > 0   詞彙表大小(Size of the vocabulary, i.e. maximum integer index + 1)  
    output_dim,    int >= 0. Dimension of the dense embedding
    
    embeddings_initializer='uniform',  Initializer for the embeddings matrix.
    embeddings_regularizer=None,       Regularizer function applied to the embeddings matrix.
    activity_regularizer=None,   
    embeddings_constraint=None,        約束函數Constraint function applied to the embeddings matrix.
    mask_zero=False, 
    input_length=None,     輸入序列的長度Length of input sequences, when it is constant. 
                           如果你需要連接 Flatten 和 Dense 層，則這個參數是必須的 
                           沒有它，dense 層的輸出尺寸就無法計算
    **kwargs
)

Input shape: 2D tensor with shape: (batch_size, input_length).
Output shape: 3D tensor with shape: (batch_size, input_length, output_dim).
```
```
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))

# the model will take as input an integer matrix of size (batch,input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```
# 硬寫 Embedding layer   Learning embeddings from scratch
```
使用imdb資料集
```
# 載入資料集與預處理(Pre-processing)
```
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
encoder.subwords[:20]

train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)

train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()
```
# 構建模型
```
embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()
```
# Compile the model
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

# 訓練模型train the model
```
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)
```

# 建立 準確率（accuracy）和損失值（loss）隨時間變化的圖表
```
import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()
```

# Retrieve the learned embeddings
```
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
```
