#
```

TensorFlow 2 — BERT: Movie Review Sentiment Analysis
```

```
# Install the required package
!pip install bert-for-tf2

from google.colab import files
uploaded = files.upload()
```
```
# -*- coding: utf-8 -*-

# Import modules
import pandas as pd
import numpy as np
import bert
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
import matplotlib.pyplot as plt

print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)
pd.set_option('display.max_colwidth',1000)

"""## Data preprocessing"""

# Read the IMDB Dataset.csv into Pandas dataframe
df=pd.read_csv('IMDB Dataset.csv')

# Take a peek at the dataset
df.head(5)

print("The number of rows and columns in the dataset is: {}".format(df.shape))

# Identify missing values
df.apply(lambda x: sum(x.isnull()), axis=0)

# Check the target class balance
df["sentiment"].value_counts()

# Functions for constructing BERT Embeddings: input_ids, input_masks, input_segments and Inputs
MAX_SEQ_LEN=500 # max sequence length

def get_masks(tokens):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1]*len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))
 
def get_segments(tokens):
    """Segments: 0 for the first sequence, 1 for the second"""  
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def get_ids(tokens, tokenizer):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:max_len]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
 
    ids = get_ids(stokens, tokenizer)
    masks = get_masks(stokens)
    segments = get_segments(stokens)

    return ids, masks, segments
 
def convert_sentences_to_features(sentences, tokenizer):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
 
    for sentence in tqdm(sentences,position=0, leave=True):
      ids,masks,segments=create_single_input(sentence,tokenizer,MAX_SEQ_LEN-2)
      assert len(ids) == MAX_SEQ_LEN
      assert len(masks) == MAX_SEQ_LEN
      assert len(segments) == MAX_SEQ_LEN
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 
          np.asarray(input_masks, dtype=np.int32), 
          np.asarray(input_segments, dtype=np.int32)]

def create_tonkenizer(bert_layer):
    """Instantiate Tokenizer with vocab"""
    vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case=bert_layer.resolved_object.do_lower_case.numpy() 
    tokenizer=bert.bert_tokenization.FullTokenizer(vocab_file,do_lower_case)
    return tokenizer

"""## Modelling"""

def nlp_model(callable_object):
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=callable_object, trainable=True)  
   
    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")           
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")       
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")
    
    inputs = [input_ids, input_masks, input_segments] # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs) # BERT outputs
    
    # Add a hidden layer
    x = Dense(units=768, activation='relu')(pooled_output)
    x = Dropout(0.1)(x)
 
    # Add output layer
    outputs = Dense(2, activation="softmax")(x)

    # Construct a new model
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = nlp_model("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
model.summary()

"""## Model training"""

# Create examples for training and testing
df = df.sample(frac=1) # Shuffle the dataset
tokenizer = create_tonkenizer(model.layers[3])
X_train = convert_sentences_to_features(df['review'][:40000], tokenizer)
X_test = convert_sentences_to_features(df['review'][40000:], tokenizer)

df['sentiment'].replace('positive',1.,inplace=True)
df['sentiment'].replace('negative',0.,inplace=True)
one_hot_encoded = to_categorical(df['sentiment'].values)
y_train = one_hot_encoded[:40000]
y_test =  one_hot_encoded[40000:]

# Train the model
BATCH_SIZE = 8
EPOCHS = 1

# Use Adam optimizer to minimize the categorical_crossentropy loss
opt = Adam(learning_rate=2e-5)
model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Fit the data to the model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose = 1)

# Save the trained model
model.save('nlp_model.h5')

"""## Analysis of model performance"""

# Load the pretrained nlp_model
from tensorflow.keras.models import load_model
new_model = load_model('nlp_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# Predict on test dataset
from sklearn.metrics import classification_report
pred_test = np.argmax(new_model.predict(X_test), axis=1)

print(classification_report(np.argmax(y_test,axis=1), pred_test))

pred_test[:10]

# Predicted 0 for the first review in the test dataset
df['review'][40000:40001]

# Predicted 1 for the second review in the test dataset
df['review'][40001:40002]

# Predicted 1 for the third review in the test dataset
df['review'][40002:40003]
```
