#
```

```

# 英文自然語言處理的經典工具: NLTK
```
英文自然語言處理的經典工具: NLTK
2019-07-30 ccs96307

 NLTK 進行文本前處理的幾個流程項目：
sentence segmentation (斷句)
word segmentation (斷詞)
pos (詞性標記)
lemmatization (字型還原)
stopword (停用詞)
ner (命名實體辨識)

https://clay-atlas.com/blog/2019/07/30/%E3%80%90nlp%E3%80%91nltk-%E5%9F%BA%E6%9C%AC%E6%95%99%E5%AD%B8/
```
### sentence segmentation (斷句)
```
nltk.download('punkt')
```

```
import nltk

text = """I went to Japan. (NOT I went to the Japan.)
He played tennis with Ben. (NOT He played tennis with the Ben.)
They had breakfast at 9 o’clock. (NOT They had a breakfast at 9 o'clock.)
(Some words don't have an article. We don't usually use articles for countries, meals or people.)"""

sentences = nltk.sent_tokenize(text)
sentences
```
### word segmentation (斷詞)
```
from nltk import word_tokenize
##tokens = word_tokenize(raw)
#tokens = [tokenizer.tokenize(sent) for sent in sentences]
tokens = [word_tokenize(sent) for sent in sentences]
tokens
```
### pos (詞性標記)
```
nltk.download('averaged_perceptron_tagger')
```
```
pos = [nltk.pos_tag(token) for token in tokens]
pos
```

# Python自然語言處理 NLTK 庫用法入門教程【經典】
```
指令碼專欄 · 發表 2018-06-26
https://www.itread01.com/article/1529980432.html
```
```
from bs4 import BeautifulSoup
import urllib.request
import nltk

response = urllib.request.urlopen('http://php.net/')
html = response.read()
soup = BeautifulSoup(html,"html5lib")
text = soup.get_text(strip=True)

tokens = [t for t in text.split()]
freq = nltk.FreqDist(tokens)
for key,val in freq.items():
  print (str(key) + ':' + str(val))
```
# Corpus語料庫 and WordNet
```
# 存取NLTK內建的語料庫Corpus
from nltk.corpus import reuters

files = reuters.fileids()
print(files)

words16097 = reuters.words(['test/16097'])
print(words16097)

words20 = reuters.words(['test/16097'])[:20]
print(words20)


reutersGenres = reuters.categories()
print(reutersGenres)


for w in reuters.words(categories=['bop','cocoa']):
    print(w+' ',end='')
    if(w is '.'):
        print()
```
```
# 1.4　計算布朗語料庫中三種不同類別的特殊疑問詞

import nltk
from nltk.corpus import brown

print(brown.categories())

genres = ['fiction', 'humor', 'romance']
whwords = ['what', 'which', 'how', 'why', 'when', 'where', 'who']


for i in range(0,len(genres)):
    genre = genres[i]
    print()
    print("Analysing '"+ genre + "' wh words")
    genre_text = brown.words(categories = genre)
    fdist = nltk.FreqDist(genre_text)
    for wh in whwords:
        print(wh + ':', fdist[wh], end=' ')
```
```
# 1-5探討網路文本和聊天文本的詞頻分佈

import nltk
nltk.download('webtext')

from nltk.corpus import webtext
print(webtext.fileids())

fileid = 'singles.txt'
wbt_words = webtext.words(fileid)
fdist = nltk.FreqDist(wbt_words)

print('Count of the maximum appearing word "',fdist.max(),'" : ', fdist[fdist.max()])
print('Total Number of distinct tokens in the bag : ', fdist.N())
print('Following are the most common 10 words in the bag')
print(fdist.most_common(10))
print('Frequency Distribution on Personal Advertisements')
print(fdist.tabulate())
fdist.plot(cumulative=True)
```
# WordNet
```
https://zh.wikipedia.org/wiki/WordNet

WordNet是一個由普林斯頓大學認識科學實驗室在心理學教授喬治·A·米勒的指導下建立和維護的英語字典。
開發工作從1985年開始，從此以後該項目接受了超過300萬美元的資助。 
由於它包含了語義信息，所以有別於通常意義上的字典。
WordNet根據詞條的意義將它們分組，每一個具有相同意義的字條組稱為一個synset。

在WordNet中，名詞，動詞，形容詞和副詞各自被組織成一個同義詞的網絡，
每個同義詞集合都代表一個基本的語義概念，並且這些集合之間也由各種關係連接。
（一個多義詞將出現在它的每個意思的同義詞集合中）。
在WordNet的第一版中（標記為1.x），四種不同詞性的網絡之間並無連接。
WordNet的名詞網絡是第一個發展起來的。

名詞網絡的主幹是蘊涵關係的層次（上位／下位關係），它占據了關係中的將近80%。
層次中的最頂層是11個抽象概念，稱為基本類別始點（unique beginners），
例如實體（entity，「有生命的或無生命的具體存在」），
心理特徵（psychological feature，「生命有機體的精神上的特徵）。
名詞層次中最深的層次是16個節點。
```
```
https://zhuanlan.zhihu.com/p/26527203
```
###
```
中文詞彙網路 (Chinese Wordnet, 以下簡稱中文詞網) 計畫，
目的是在提供完整的中文詞義 (sense) 區分與詞彙語意關係知識庫
http://lope.linguistics.ntu.edu.tw/cwn/
```
# 1.6　使用WordNet進行詞義消歧
```
from nltk.corpus import wordnet as wn

woman = wn.synset('woman.n.01')
bed = wn.synset('bed.n.01')

print(woman.hypernyms())
woman_paths = woman.hypernym_paths()

for idx, path in enumerate(woman_paths):
    print('\n\nHypernym Path :', idx + 1)
    for synset in path:
        print(synset.name(), ', ', end='')



types_of_beds = bed.hyponyms()
print('\n\nTypes of beds(Hyponyms): ', types_of_beds)

print(sorted(set(lemma.name() for synset in types_of_beds for lemma in synset.lemmas())))
```
# 1.7　選擇兩個不同的同義詞集，使用WordNet探討上位詞和下位詞的概念
```
from nltk.corpus import wordnet as wn

woman = wn.synset('woman.n.01')
bed = wn.synset('bed.n.01')

print(woman.hypernyms())
woman_paths = woman.hypernym_paths()

for idx, path in enumerate(woman_paths):
    print('\n\nHypernym Path :', idx + 1)
    for synset in path:
        print(synset.name(), ', ', end='')



types_of_beds = bed.hyponyms()
print('\n\nTypes of beds(Hyponyms): ', types_of_beds)

print(sorted(set(lemma.name() for synset in types_of_beds for lemma in synset.lemmas())))
```

# 1.8　基於WordNet計算名詞、動詞、形容詞和副詞的平均多義性
```
from nltk.corpus import wordnet as wn
type = 'n'

synsets = wn.all_synsets(type)

lemmas = []
for synset in synsets:
    for lemma in synset.lemmas():
        lemmas.append(lemma.name())

print(len(lemmas))
lemmas = set(lemmas)
print('Total distinct lemmas: ', len(lemmas))

count = 0
for lemma in lemmas:
    count = count + len(wn.synsets(lemma, type))

print('Total senses :',count)
print('Average Polysemy of ', type,': ' ,  count/len(lemmas))
```
#
```
第3章　預處理
3.1　引言
3.2　分詞——學習使用NLTK內置的分詞器
3.3　詞幹提取——學習使用NLTK內置的詞幹提取器
3.4　詞形還原——學習使用NLTK中的WordnetLemmatizer函數
3.5　停用詞——學習使用停用詞語料庫及其應用
3.6　編輯距離——編寫計算兩個字串之間編輯距離的演算法
3.7　處理兩篇短文並提取共有詞彙
```
# 3.2　分詞Tokenization ——學習使用NLTK內置的分詞器inbuilt tokenizers of NLTK
```
from nltk.tokenize import LineTokenizer, SpaceTokenizer, TweetTokenizer
from nltk import word_tokenize

lTokenizer = LineTokenizer();
print("Line tokenizer output :",lTokenizer.tokenize("My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. \nFather to a murdered son, husband to a murdered wife. \nAnd I will have my vengeance, in this life or the next."))

rawText = "By 11 o'clock on Sunday, the doctor shall open the dispensary."
sTokenizer = SpaceTokenizer()
print("Space Tokenizer output :",sTokenizer.tokenize(rawText))

print("Word Tokenizer output :", word_tokenize(rawText))

tTokenizer = TweetTokenizer()
print("Tweet Tokenizer output :",tTokenizer.tokenize("This is a cooool #dummysmiley: :-) :-P <3"))
```


# 3.3　詞幹提取Stemming——學習使用NLTK內置的詞幹提取器 inbuilt stemmers of NLTK
```
from nltk import PorterStemmer, LancasterStemmer, word_tokenize

raw = "My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."

tokens = word_tokenize(raw)

porter = PorterStemmer()
pStems = [porter.stem(t) for t in tokens]
print(pStems)

lancaster = LancasterStemmer()
lStems = [lancaster.stem(t) for t in tokens]
print(lStems)
```


# 3.4　詞形還原Lemmatization——學習使用NLTK中的WordnetLemmatizer函數
```
nltk.download('gutenberg')
nltk.download('stopwords')
```
```
import nltk
from nltk.corpus import gutenberg
print(gutenberg.fileids())

gb_words = gutenberg.words('bible-kjv.txt')
words_filtered = [e.lower() for e in gb_words if len(e) >= 3]
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words_filtered if w.lower() not in stopwords]

fdist = nltk.FreqDist(words)
fdist2 = nltk.FreqDist(gb_words)

print('Following are the most common 10 words in the bag')
print(fdist2.most_common(10))
print('Following are the most common 10 words in the bag minus the stopwords')
print(fdist.most_common(10))
fdist.plot()
```


# 3.5　停用詞Stopwords ——學習使用停用詞語料庫及其應用
```



```


#  3.6　編輯距離Edit distance——編寫計算兩個字串之間編輯距離的演算法writing your own algorithm to find edit distance between two strings
```

from nltk.metrics.distance import edit_distance

def my_edit_distance(str1, str2):
    m= len(str1) + 1
    n= len(str2) + 1

    table = {}
    for i in range(m): table[i,0]=i
    for j in range(n): table[0,j]=j

    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            table[i,j] = min(table[i, j-1]+1, table[i-1, j]+1, table[i-1, j-1]+cost)

    return table[i,j]

print("Our Algorithm :",my_edit_distance("hand", "and"))
print("NLTK Algorithm :",edit_distance("hand", "and"))
```


# 3.7　處理兩篇短文並提取共有詞彙Processing two short stories and extracting the common vocabulary between two of them
```
story1 = """In a far away kingdom, there was a river. This river was home to many golden swans. The swans spent most of their time on the banks of the river. Every six months, the swans would leave a golden feather as a fee for using the lake. The soldiers of the kingdom would collect the feathers and deposit them in the royal treasury. 
One day, a homeless bird saw the river. "The water in this river seems so cool and soothing. I will make my home here," thought the bird. 
As soon as the bird settled down near the river, the golden swans noticed her. They came shouting. "This river belongs to us. We pay a golden feather to the King to use this river. You can not live here." 
"I am homeless, brothers. I too will pay the rent. Please give me shelter," the bird pleaded. "How will you pay the rent? You do not have golden feathers," said the swans laughing. They further added, "Stop dreaming and leave once." The humble bird pleaded many times. But the arrogant swans drove the bird away. 
"I will teach them a lesson!" decided the humiliated bird. 
She went to the King and said, "O King! The swans in your river are impolite and unkind. I begged for shelter but they said that they had purchased the river with golden feathers." 
The King was angry with the arrogant swans for having insulted the homeless bird. He ordered his soldiers to bring the arrogant swans to his court. In no time, all the golden swans were brought to the King’s court. 
"Do you think the royal treasury depends upon your golden feathers? You can not decide who lives by the river. Leave the river at once or you all will be beheaded!" shouted the King. 
The swans shivered with fear on hearing the King. They flew away never to return. The bird built her home near the river and lived there happily forever. The bird gave shelter to all other birds in the river. """

story2 = """Long time ago, there lived a King. He was lazy and liked all the comforts of life. He never carried out his duties as a King. “Our King does not take care of our needs. He also ignores the affairs of his kingdom." The people complained. 
One day, the King went into the forest to hunt. After having wandered for quite sometime, he became thirsty. To his relief, he spotted a lake. As he was drinking water, he suddenly saw a golden swan come out of the lake and perch on a stone. “Oh! A golden swan. I must capture it," thought the King. 
But as soon as he held his bow up, the swan disappeared. And the King heard a voice, “I am the Golden Swan. If you want to capture me, you must come to heaven." 
Surprised, the King said, “Please show me the way to heaven." “Do good deeds, serve your people and the messenger from heaven would come to fetch you to heaven," replied the voice. 
The selfish King, eager to capture the Swan, tried doing some good deeds in his Kingdom. “Now, I suppose a messenger will come to take me to heaven," he thought. But, no messenger came. 
The King then disguised himself and went out into the street. There he tried helping an old man. But the old man became angry and said, “You need not try to help. I am in this miserable state because of out selfish King. He has done nothing for his people." 
Suddenly, the King heard the golden swan’s voice, “Do good deeds and you will come to heaven." It dawned on the King that by doing selfish acts, he will not go to heaven. 
He realized that his people needed him and carrying out his duties was the only way to heaven. After that day he became a responsible King. 
"""

story1 = story1.replace(",", "").replace("\n", "").replace('.', '').replace('"', '').replace("!","").replace("?","").casefold()
story2 = story2.replace(",", "").replace("\n", "").replace('.', '').replace('"', '').replace("!","").replace("?","").casefold()

story1_words = story1.split(" ")
print("First Story words :",story1_words)
story2_words = story2.split(" ")
print("Second Story words :",story2_words)

story1_vocab = set(story1_words)
print("First Story vocabulary :",story1_vocab)
story2_vocab = set(story2_words)
print("Second Story vocabulary",story2_vocab)

common_vocab = story1_vocab & story2_vocab
print("Common Vocabulary :",common_vocab)
```
```
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

Following are the most common 10 words in the bag
[(',', 70509), ('the', 62103), (':', 43766), ('and', 38847), ('of', 34480), ('.', 26160), ('to', 13396), ('And', 12846), ('that', 12576), ('in', 12331)]

Following are the most common 10 words in the bag minus the stopwords
[('shall', 9838), ('unto', 8997), ('lord', 7964), ('thou', 5474), ('thy', 4600), ('god', 4472), ('said', 3999), ('thee', 3827), ('upon', 2748), ('man', 2735)]

```

# 作業:完成文章的練習
```
Python自然語言處理 NLTK 庫用法入門教程【經典】
指令碼專欄 · 發表 2018-06-26
https://www.itread01.com/article/1529980432.html
```

```
NLTK Python Tutorial (Natural Language Toolkit)
https://data-flair.training/blogs/nltk-python-tutorial/
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


