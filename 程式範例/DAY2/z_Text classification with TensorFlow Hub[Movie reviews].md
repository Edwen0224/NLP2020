#
```



```

```
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

!pip install tensorflow-hub
!pip install tfds-nightly
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Download the IMDB dataset
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

"""## Explore the data 

Let's take a moment to understand the format of the data. Each example is a sentence representing the movie review and a corresponding label. The sentence is not preprocessed in any way. The label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

Let's print first 10 examples.
"""

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

"""Let's also print the first 10 labels."""

train_labels_batch

"""## Build the model
create a Keras layer that uses a TensorFlow Hub model to embed the sentences, 
and try it out on a couple of input examples. 

Note that no matter the length of the input text, 
the output shape of the embeddings is: `(num_examples, embedding_dimension)`.
"""

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

"""build the model:"""

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

"""The layers are stacked sequentially to build the classifier:

1. The first layer is a TensorFlow Hub layer. 
This layer uses a pre-trained Saved Model to map a sentence into its embedding vector. 
The pre-trained text embedding model that we are using 
([google/tf2-preview/gnews-swivel-20dim/1]
(https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)) 
splits the sentence into tokens, embeds each token and then combines the embedding. 
The resulting dimensions are: `(num_examples, embedding_dimension)`.
2. This fixed-length output vector is piped through a fully-connected (`Dense`) layer with 16 hidden units.
3. The last layer is densely connected with a single output node.
"""
#compile the model.
"""
Loss function and optimizer
A model needs a loss function and an optimizer for training. 

loss function
Since this is a binary classification problem 
and the model outputs logits (a single-unit layer with a linear activation), 
we'll use the `binary_crossentropy` loss function.

This isn't the only choice for a loss function, you could, for instance, 
choose `mean_squared_error`. 
But, generally, `binary_crossentropy` is better for dealing with probabilities—it 
measures the "distance" between probability distributions, 
or in our case, between the ground-truth distribution and the predictions.
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""## Train the model

Train the model for 20 epochs in mini-batches of 512 samples. 
This is 20 iterations over all samples in the `x_train` and `y_train` tensors. 
While training, monitor the model's loss and accuracy on the 10,000 samples 
from the validation set
"""

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

"""## Evaluate the model
see how the model performs. 
Two values will be returned:
Loss (a number which represents our error, lower values are better), and accuracy.
"""

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

"""
This fairly naive approach achieves an accuracy of about 87%. 
With more advanced approaches, the model should get closer to 95%.
"""
```
```
49/49 - 3s - loss: 0.3188 - accuracy: 0.8539
loss: 0.319
accuracy: 0.854
```
