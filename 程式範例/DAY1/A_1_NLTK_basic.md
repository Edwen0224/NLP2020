#
```



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
```
1.6　使用WordNet進行詞義消歧
1.7　選擇兩個不同的同義詞集，使用WordNet探討上位詞和下位詞的概念
1.8　基於WordNet計算名詞、動詞、形容詞和副詞的平均多義性
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


#
```



```


