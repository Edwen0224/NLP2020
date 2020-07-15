# ELMo(2018)
```
Deep contextualized word representations
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer
https://arxiv.org/abs/1802.05365
```
# 案例分析
```
ELMo for IMDB Movie Review Classification
https://www.kaggle.com/azunre/tl-for-nlp-section2-5-movies-elmo
```

```
Using ELMo for "fake news" detection/classification
https://www.kaggle.com/azunre/tl-for-nlp-sections4-2-4-4-fake-news-elmo
```
# 案例分析1:ELMo for IMDB Movie Review Classification
```

# Preliminaries
# First install a critical dependency for our code

!pip install keras==2.2.4 # critical dependency

# Write requirements to file, anytime you run it, in case you have to go back and recover dependencies.
"""
Latest known such requirements are hosted for each notebook in the companion github repo, 
and can be pulled down and installed here if needed. 
Companion github repo is located at https://github.com/azunre/transfer-learning-for-nlp
"""

!pip freeze > kaggle_image_requirements.txt

# Import neural network libraries
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer

# Initialize tensorflow/keras session
sess = tf.Session()
K.set_session(sess)


# Some other key imports
import os
import re
import pandas as pd
import numpy as np
import random

# Define Tokenization, Stop-word and Punctuation Removal Functions
"""
Before proceeding, we must decide how many samples to draw from each class. 
We must also decide the maximum number of tokens per email, and the maximum length of each token. 
This is done by setting the following overarching hyperparameters
"""

Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 50 # the maximum number of tokens per document
maxtokenlen = 20 # the maximum length of each token


# Tokenization

def tokenize(row):
    if row is None or row is '':
        tokens = ""
    else:
        tokens = row.split(" ")[:maxtokens]
    return tokens

# Use regular expressions to remove unnecessary characters
"""
Next, we define a function to remove punctuation marks and other nonword characters
(using regular expressions) from the emails with the help of the ubiquitous python regex library. 
In the same step, we truncate all tokens to hyperparameter maxtokenlen defined above.
"""

import re

def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokenlen] # truncate token
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens
    
# Stop-word removal
"""
Let’s define a function to remove stopwords - 
words that occur so frequently in language that they offer no useful information for classification. 

This includes words such as “the” and “are”, and the popular library NLTK provides a heavily-used list that will employ.
"""

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')    

# print(stopwords) # see default stopwords
# it may be beneficial to drop negation words from the removal list, as they can change the positive/negative meaning
# of a sentence - but we didn't find it to make a difference for this problem
# stopwords.remove("no")
# stopwords.remove("nor")
# stopwords.remove("not")


def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token

# Download and Assemble IMDB Review Dataset
# Download the labeled IMDB reviews

!wget -q "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
!tar xzf aclImdb_v1.tar.gz


# Shuffle and preprocess data
# function for shuffling data
def unison_shuffle(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header

def load_data(path):
    data, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), 'r') as reader:
                  text = reader.read()
            text = tokenize(text)
            text = stop_word_removal(text)
            text = reg_expressions(text)
            data.append(text)
            sentiments.append(sentiment)
    data_np = np.array(data)
    data, sentiments = unison_shuffle(data_np, sentiments)
    
    return data, sentiments

train_path = os.path.join('aclImdb', 'train')
test_path = os.path.join('aclImdb', 'test')
raw_data, raw_header = load_data(train_path)

print(raw_data.shape)
print(len(raw_header))

# Subsample required number of samples
random_indices = np.random.choice(range(len(raw_header)),size=(Nsamp*2,),replace=False)
data_train = raw_data[random_indices]
header = raw_header[random_indices]

print("DEBUG::data_train::")
print(data_train)

# Display sentiments and their frequencies in the dataset, to ensure it is roughly balanced between classes

unique_elements, counts_elements = np.unique(header, return_counts=True)
print("Sentiments and their frequencies:")
print(unique_elements)
print(counts_elements)


# function for converting data into the right format, due to the difference in required format from sklearn models
# we expect a single string per email here, versus a list of tokens for the sklearn models previously explored

def convert_data(raw_data,header):
    converted_data, labels = [], []
    for i in range(raw_data.shape[0]):
        # combine list of tokens representing each email into single string
        out = ' '.join(raw_data[i])
        converted_data.append(out)
        labels.append(header[i])
    converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]
    
    return converted_data, np.array(labels)

data_train, header = unison_shuffle(data_train, header)

# split into independent 70% training and 30% testing sets
idx = int(0.7*data_train.shape[0])

# 70% of data for training
train_x, train_y = convert_data(data_train[:idx],header[:idx])

# remaining 30% for testing
test_x, test_y = convert_data(data_train[idx:],header[idx:])

print("train_x/train_y list details, to make sure it is of the right form:")
print(len(train_x))
print(train_x)
print(train_y[:5])
print(train_y.shape)

# Build, Train and Evaluate ELMo Model

# Create a custom tf hub ELMO embedding layer

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024 # initialize output dimension of ELMo embedding
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape): # function for building ELMo embedding
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name)) # download pretrained ELMo model
        # extract trainable parameters, which are only a small subset of the total - this is a constraint of
        # the tf hub module as shared by the authors - see https://tfhub.dev/google/elmo/2
        # the trainable parameters are 4 scalar weights on the sum of the outputs of ELMo layers
        
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None): # specify function for calling embedding
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_output_shape(self, input_shape): # specify output shape
        return (input_shape[0], self.dimensions)
        
# We now use the custom TF hub ELMo embedding layer within a higher-level function to define the overall model. 
# More specifically, we put a dense trainable layer of output dimension 256 on top of the ELMo embedding.


# Function to build model
def build_model(): 
    input_text = layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


# Build and fit
model = build_model()
history = model.fit(train_x, 
          train_y,
          validation_data=(test_x, test_y),
          epochs=5,
          batch_size=32)

# Save trained model
model.save('ElmoModel.h5')

# Visualize Convergence


import matplotlib.pyplot as plt

df_history = pd.DataFrame(history.history)

fig,ax = plt.subplots()
plt.plot(range(df_history.shape[0]),df_history['val_acc'],'bs--',label='validation')
plt.plot(range(df_history.shape[0]),df_history['acc'],'r^--',label='training')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('ELMo Movie Review Classification Training')
plt.legend(loc='best')
plt.grid()
plt.show()

fig.savefig('ELMoConvergence.eps', format='eps')
fig.savefig('ELMoConvergence.pdf', format='pdf')
fig.savefig('ELMoConvergence.png', format='png')
fig.savefig('ELMoConvergence.svg', format='svg')

# Make figures downloadable to local system in interactive mode

from IPython.display import HTML
def create_download_link(title = "Download file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

create_download_link(filename='ELMoConvergence.svg')


# you must remove all downloaded files - having too many of them on completion will make Kaggle reject your notebook 
!rm -rf aclImdb
!rm aclImdb_v1.tar.gz
```
# 案例分析2:假新聞偵測
```
Using ELMo for "fake news" detection/classification
https://www.kaggle.com/azunre/tl-for-nlp-sections4-2-4-4-fake-news-elmo
```


```
# Preliminaries
# First install a critical dependency for our code

!pip install keras==2.2.4 # critical dependency, 


# Write requirements to file, anytime you run it, in case you have to go back and recover dependencies.
"""
Latest known such requirements are hosted for each notebook in the companion github repo, 
and can be pulled down and installed here if needed. 
Companion github repo is located at https://github.com/azunre/transfer-learning-for-nlp
"""

!pip freeze > kaggle_image_requirements.txt

# Read and Preprocess Fake News Dataset
# First let's list the available data files

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

Read in the "true" and "fake" data
"""
In quotes, because that has the potential to simply replicate the biases of the labeler, so should be carefully evaluated
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the data into pandas DataFrames
DataTrue = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
DataFake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

print("Data labeled as True:")
print(DataTrue.head())
print("\n\n\nData labeled as Fake:")
print(DataFake.head())

# Assemble the two different kinds of data

Nsamp =1000 # number of samples to generate in each class - 'true', 'fake'
DataTrue = DataTrue.sample(Nsamp)
DataFake = DataFake.sample(Nsamp)
raw_data = pd.concat([DataTrue,DataFake], axis=0).values

# combine title, body text and topics into one string per document
raw_data = [sample[0].lower() + sample[1].lower() + sample[3].lower() for sample in raw_data]

print("Length of combined data is:")
print(len(raw_data))
print("Data represented as numpy array (first 5 samples) is:")
print(raw_data[:5])

# corresponding labels
Categories = ['True','False']
header = ([1]*Nsamp)
header.extend(([0]*Nsamp))


# Shuffle data, split into train and test sets...

# function for shuffling data
def unison_shuffle(a, b):
    p = np.random.permutation(len(b))
    data = np.asarray(a)[p]
    header = np.asarray(b)[p]
    return data, header

raw_data, header = unison_shuffle(raw_data, header)

# split into independent 70% training and 30% testing sets
idx = int(0.7*raw_data.shape[0])

# 70% of data for training
train_x = raw_data[:idx]
train_y = header[:idx]
# remaining 30% for testing
test_x = raw_data[idx:]
test_y = header[idx:]

print("train_x/train_y list details, to make sure it is of the right form:")
print(len(train_x))
#print(train_x)
print(train_y[:5])
print(train_y.shape)

# Build, Train and Evaluate ELMo Model
# First import required neural network libraries

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer

# Initialize tensorflow/keras session
sess = tf.Session()
K.set_session(sess)


# Create a custom tf hub ELMO embedding layer

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024 # initialize output dimension of ELMo embedding
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)


    def build(self, input_shape): # function for building ELMo embedding
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name)) # download pretrained ELMo model
        
        # extract trainable parameters, which are only a small subset of the total - this is a constraint of
        # the tf hub module as shared by the authors - see https://tfhub.dev/google/elmo/2
        # the trainable parameters are 4 scalar weights on the sum of the outputs of ELMo layers 
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None): # specify function for calling embedding
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_output_shape(self, input_shape): # specify output shape
        return (input_shape[0], self.dimensions)

"""
We now use the custom TF hub ELMo embedding layer within a higher-level function to define the overall model. 
More specifically, we put a dense trainable layer of output dimension 256 on top of the ELMo embedding.
"""

# Function to build overall model
def build_model():
    input_text = layers.Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(1, activation='sigmoid')(dense)    
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    return model

print(train_x.shape)

# Build and fit
model = build_model()
history = model.fit(train_x, 
          train_y,
          validation_data=(test_x, test_y),
          epochs=10,
          batch_size=4)

# Save trained model
model.save('ELMoModel.h5')

# Visualize Convergence

import matplotlib.pyplot as plt

df_history = pd.DataFrame(history.history)
fig,ax = plt.subplots()
plt.plot(range(df_history.shape[0]),df_history['val_acc'],'bs--',label='validation')
plt.plot(range(df_history.shape[0]),df_history['acc'],'r^--',label='training')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('ELMo Fake News Classification Training')
plt.legend(loc='best')
plt.grid()
plt.show()

# Save figures
fig.savefig('ELMoConvergence.eps', format='eps')
fig.savefig('ELMoConvergence.pdf', format='pdf')
fig.savefig('ELMoConvergence.png', format='png')
fig.savefig('ELMoConvergence.svg', format='svg')


# Make figures downloadable to local system in interactive mode

from IPython.display import HTML
def create_download_link(title = "Download file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

create_download_link(filename='ELMoConvergence.svg')

```
