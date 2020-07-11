# 文本分類Text classification::電影評論
```
https://www.tensorflow.org/tutorials/keras/text_classification
```



# 環境設定Setup
```
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

```
# Download the IMDB dataset  下載 IMDB 資料集
```
IMDB 資料集已經打包在 Tensorflow `tfds` 中。
該資料集已經經過預處理，
評論（單詞序列）已經被轉換為整數序列，其中每個整數表示字典中的特定單詞。

The IMDB movie reviews dataset comes packaged in `tfds`. 
It has already been preprocessed so that the reviews (sequences of words) have been converted to sequences of integers, 
where each integer represents a specific word in a dictionary.

The following code downloads the IMDB dataset to your machine (or uses a cached copy if you've already downloaded it):
To encode your own text see the [Loading text tutorial](../load_data/text.ipynb)
```
```
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)

```

## Try the encoder
```
"""
 The dataset `info` includes the text encoder (a `tfds.features.text.SubwordTextEncoder`).
 
 tfds.features.text.SubwordTextEncoder
 https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
"""

encoder = info.features['text'].encoder

print ('Vocabulary size: {}'.format(encoder.vocab_size))

"""This text encoder will reversibly encode any string:"""

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

"""
The encoder encodes the string by breaking it into subwords or characters if the word is not in its dictionary. 
So the more a string resembles the dataset, the shorter the encoded representation will be."""

for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

"""
```
# Explore the data
```
Let's take a moment to understand the format of the data. 
The dataset comes preprocessed: each example is an array of integers representing the words of the movie review. 
The text of reviews have been converted to integers, where each integer represents a specific word-piece in the dictionary. 
Each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.
Here's what the first review looks like:
"""

for train_example, train_label in train_data.take(1):
  print('Encoded text:', train_example[:10].numpy())
  print('Label:', train_label.numpy())

"""The `info` structure contains the encoder/decoder. The encoder can be used to recover the original text:"""

encoder.decode(train_example)

```
# Prepare the data for training
```

BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32))

test_batches = (
    test_data
    .padded_batch(32))

"""Each batch will have a shape of `(batch_size, sequence_length)` because the padding is dynamic each batch will have a different length:"""

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

```
# Build the model
```
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1)])

model.summary()

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

```
# Train the model
```

history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

```
# Evaluate the model
```
loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


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

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```
