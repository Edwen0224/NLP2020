#
```
Neural machine translation with attention
https://www.tensorflow.org/tutorials/text/nmt_with_attention
```

```
# -*- coding: utf-8 -*-


"""# 基於注意力的神經機器翻譯

此筆記本訓練一個將西班牙語翻譯為英語的序列到序列（sequence to sequence，簡寫為 seq2seq）模型。
此例子難度較高，需要對序列到序列模型的知識有一定瞭解。

訓練完此筆記本中的模型後，你將能夠輸入一個西班牙語句子，
例如 *"¿todavia estan en casa?"*，並返回其英語翻譯 *"are you still at home?"*

對於一個簡單的例子來說，翻譯品質令人滿意。但是更有趣的可能是生成的注意力圖：
它顯示在翻譯過程中，輸入句子的哪些部分受到了模型的注意。

<img src="https://tensorflow.google.cn/images/spanish-english.png" alt="spanish-english attention plot">

請注意：運行這個例子用一個 P100 GPU 需要花大約 10 分鐘。
"""

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

"""## 下載和準備資料集

我們將使用 http://www.manythings.org/anki/ 提供的一個語言資料集。

這個資料集包含如下格式的語言翻譯對：

```
May I borrow this book?	¿Puedo tomar prestado este libro?
```

這個資料集中有很多種語言可供選擇。我們將使用英語 - 西班牙語資料集。
為方便使用，我們在穀歌雲上提供了此資料集的一份副本。但是你也可以自己下載副本。
下載完資料集後，我們將採取下列步驟準備資料：

1. 給每個句子添加一個 *開始* 和一個 *結束* 標記（token）。
2. 刪除特殊字元以清理句子。
3. 創建一個單詞索引和一個反向單詞索引（即一個從單詞映射至 id 的詞典和一個從 id 映射至單詞的詞典）。
4. 將每個句子填充（pad）到最大長度。
"""

# 下載檔案
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# 將 unicode 檔轉換為 ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # 在單詞與跟在其後的標點符號之間插入一個空格
    # 例如： "he is a boy." => "he is a boy ."
    # 參考：https://stackoverflow.com/questions/3645931/
    # python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，將所有字元替換為空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # 給句子加上開始和結束標記
    # 以便模型知道何時開始和結束預測
    w = '<start> ' + w + ' <end>'
    return w

en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

# 1. 去除重音符號
# 2. 清理句子
# 3. 返回這樣格式的單詞對：[ENGLISH, SPANISH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # 創建清理過的輸入輸出對
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

"""
### 限制資料集的大小以加快實驗速度（可選）
在超過 10 萬個句子的完整資料集上訓練需要很長時間。
為了更快地訓練，我們可以將資料集的大小限制為 3 萬個句子（當然，翻譯品質也會隨著資料的減少而降低）：
"""

# 嘗試實驗不同大小的資料集
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# 計算目標張量的最大長度 （max_length）
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# 採用 80 - 20 的比例切分訓練集和驗證集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# 顯示長度
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

"""### 創建一個 tf.data 資料集"""

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

"""
## 編寫編碼器 （encoder） 和解碼器 （decoder） 模型

實現一個基於注意力的編碼器 - 解碼器模型。
關於這種模型，你可以閱讀 TensorFlow 的 [神經機器翻譯 (序列到序列) 教程](https://github.com/tensorflow/nmt)。
本示例採用一組更新的 API。此筆記本實現了上述序列到序列教程中的 [注意力方程式]
(https://github.com/tensorflow/nmt#background-on-the-attention-mechanism)。
下圖顯示了注意力機制為每個輸入單詞分配一個權重，然後解碼器將這個權重用於預測句子中的下一個單詞。
下圖和公式是 [Luong 的論文](https://arxiv.org/abs/1508.04025v5)中注意力機制的一個例子。

<img src="https://tensorflow.google.cn/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">

輸入經過編碼器模型，編碼器模型為我們提供形狀為 *(批大小，最大長度，隱藏層大小)* 的編碼器輸出
和形狀為 *(批大小，隱藏層大小)* 的編碼器隱藏層狀態。

下面是所實現的方程式：

<img src="https://tensorflow.google.cn/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
<img src="https://tensorflow.google.cn/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">

本教程的編碼器採用 [Bahdanau 注意力](https://arxiv.org/pdf/1409.0473.pdf)。

在用簡化形式編寫之前，讓我們先決定符號：
* FC = 完全連接（密集）層
* EO = 編碼器輸出
* H = 隱藏層狀態
* X = 解碼器輸入

以及偽代碼：

* `score = FC(tanh(FC(EO) + FC(H)))`
* `attention weights = softmax(score, axis = 1)`。 Softmax 默認被應用於最後一個軸，
但是這裡我們想將它應用於 *第一個軸*, 因為分數 （score） 的形狀是 *(批大小，最大長度，隱藏層大小)*。
最大長度 （`max_length`） 是我們的輸入的長度。
因為我們想為每個輸入分配一個權重，所以 softmax 應該用在這個軸上。
* `context vector = sum(attention weights * EO, axis = 1)`。選擇第一個軸的原因同上。
* `embedding output` = 解碼器輸入 X 通過一個嵌入層。
* `merged vector = concat(embedding output, context vector)`
* 此合併後的向量隨後被傳送到 GRU

每個步驟中所有向量的形狀已在代碼的注釋中闡明：
"""

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# 樣本輸入
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隱藏層的形狀 == （批大小，隱藏層大小）
    # hidden_with_time_axis 的形狀 == （批大小，1，隱藏層大小）
    # 這樣做是為了執行加法以計算分數  
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分數的形狀 == （批大小，最大長度，1）
    # 我們在最後一個軸上得到 1， 因為我們把分數應用於 self.V
    # 在應用 self.V 之前，張量的形狀是（批大小，最大長度，單位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力權重 （attention_weights） 的形狀 == （批大小，最大長度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之後的形狀 == （批大小，隱藏層大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 用於注意力
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # 編碼器輸出 （enc_output） 的形狀 == （批大小，最大長度，隱藏層大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x 在通過嵌入層後的形狀 == （批大小，1，嵌入維度）
    x = self.embedding(x)

    # x 在拼接 （concatenation） 後的形狀 == （批大小，1，嵌入維度 + 隱藏層大小）
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 將合併後的向量傳送到 GRU
    output, state = self.gru(x)

    # 輸出的形狀 == （批大小 * 1，隱藏層大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 輸出的形狀 == （批大小，vocab）
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

"""## 定義優化器和損失函數"""

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

"""## 檢查點（基於物件保存）"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

"""
## 訓練
1. 將 *輸入* 傳送至 *編碼器*，編碼器返回 *編碼器輸出* 和 *編碼器隱藏層狀態*。
2. 將編碼器輸出、編碼器隱藏層狀態和解碼器輸入（即 *開始標記*）傳送至解碼器。
3. 解碼器返回 *預測* 和 *解碼器隱藏層狀態*。
4. 解碼器隱藏層狀態被傳送回模型，預測被用於計算損失。
5. 使用 *教師強制 （teacher forcing）* 決定解碼器的下一個輸入。
6. *教師強制* 是將 *目標詞* 作為 *下一個輸入* 傳送至解碼器的技術。
7. 最後一步是計算梯度，並將其應用於優化器和反向傳播。
"""

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # 教師強制 - 將目標詞作為下一個輸入
    for t in range(1, targ.shape[1]):
      # 將編碼器輸出 （enc_output） 傳送至解碼器
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # 使用教師強制
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  # 每 2 個週期（epoch），保存（檢查點）一次模型
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

"""## 翻譯

* 評估函數類似於訓練迴圈，不同之處在於在這裡我們不使用 *教師強制*。
  每個時間步的解碼器輸入是其先前的預測、隱藏層狀態和編碼器輸出。
* 當模型預測 *結束標記* 時停止預測。
* 存儲 *每個時間步的注意力權重*。

請注意：對於一個輸入，編碼器輸出僅計算一次。
"""

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # 存儲注意力權重以便後面製圖
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 預測的 ID 被輸送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

# 注意力權重製圖函數
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))

"""## 恢復最新的檢查點並驗證"""

# 恢復檢查點目錄 （checkpoint_dir） 中最新的檢查點
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')

translate(u'esta es mi vida.')

translate(u'¿todavia estan en casa?')

# 錯誤的翻譯
translate(u'trata de averiguarlo.')
```
