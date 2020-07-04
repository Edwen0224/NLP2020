# 電影評論文本分類
```
https://www.tensorflow.org/tutorials/keras/text_classification
```
```
電影評論文本分類 Text classification with preprocessed text: Movie reviews

This notebook classifies movie reviews as *positive* or *negative* using the text of the review. 
This is an example of *binary*—or two-class—classification, 
an important and widely applicable kind of machine learning problem.
此筆記本（notebook）使用評論文本將影評分為*積極（positive）*或*消極（nagetive）*兩類。
這是一個*二元（binary）*或者二分類問題，一種重要且應用廣泛的機器學習問題。

我們將使用來源於[網路電影資料庫（Internet Movie Database）]的 IMDB 資料集（IMDB dataset），其包含 50,000 條影評文本。
從該資料集切割出的25,000條評論用作訓練，
另外 25,000 條用作測試。訓練集與測試集是*平衡的（balanced）*，
意味著它們包含相等數量的積極和消極評論。

We'll use the [IMDB dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) 
that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/). 
These are split into 25,000 reviews for training and 25,000 reviews for testing. 
The training and testing sets are *balanced*, meaning they contain an equal number of positive and negative reviews.
```
```
此notebook使用tf.keras，它是一個 Tensorflow 中用於構建和訓練模型的高級API。
This notebook uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in TensorFlow. 


For a more advanced text classification tutorial using `tf.keras`, 
see the [MLCC Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification/).
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
```
## Download the IMDB dataset  下載 IMDB 資料集

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
## Explore the data

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

"""
## Prepare the data for training
You will want to create batches of training data for your model. 
The reviews are all different lengths, so use `padded_batch` to zero pad the sequences while batching:
"""

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

"""
## Build the model

The neural network is created by stacking layers—this requires two main architectural decisions:

* How many layers to use in the model?
* How many *hidden units* to use for each layer?

In this example, the input data consists of an array of word-indices. 
The labels to predict are either 0 or 1. Let's build a "Continuous bag of words" style model for this problem:

Caution: This model doesn't use masking, so the zero-padding is used as part of the input, 
so the padding length may affect the output.  
To fix this, see the [masking and padding guide](../../guide/keras/masking_and_padding.ipynb).
"""

model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1)])

model.summary()

"""
The layers are stacked sequentially to build the classifier:
1. The first layer is an `Embedding` layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: `(batch, sequence, embedding)`.  *To learn more about embeddings, see the [word embedding tutorial](../text/word_embeddings.ipynb).*

2. Next, a `GlobalAveragePooling1D` layer returns a fixed-length output vector for each example by averaging over the sequence dimension. 
This allows the model to handle input of variable length, in the simplest way possible.

3. This fixed-length output vector is piped through a fully-connected (`Dense`) layer with 16 hidden units.

4. The last layer is densely connected with a single output node. 
This uses the default *linear* activation function that outputs *logits* for numerical stability. 
Another option is to use the *sigmoid* activation function that 
returns a float value between 0 and 1, representing a probability, or confidence level.

### Hidden units

The above model has two intermediate or "hidden" layers, between the input and output. The number of outputs (units, nodes, or neurons) is the dimension of the representational space for the layer. In other words, the amount of freedom the network is allowed when learning an internal representation.

If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to learning unwanted patterns—patterns that improve performance on training data but not on the test data. This is called *overfitting*, and we'll explore it later.

### Loss function and optimizer

A model needs a loss function and an optimizer for training. Since this is a binary classification problem and the model outputs logits (a single-unit layer with a linear activation), we'll use the `binary_crossentropy` loss function.

This isn't the only choice for a loss function, you could, for instance, choose `mean_squared_error`. But, generally, `binary_crossentropy` is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions.

Later, when we are exploring regression problems (say, to predict the price of a house), we will see how to use another loss function called mean squared error.

Now, configure the model to use an optimizer and a loss function:
"""

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
## Train the model
Train the model by passing the `Dataset` object to the model's fit function. 
Set the number of epochs.
"""

history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

"""
## Evaluate the model
"""

loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


history_dict = history.history
history_dict.keys()

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
