#
```

```
```
import json 	# 匯入 json 套件

jsonFile = open('json_test_data.txt', 'r', encoding = 'utf-8')  # 開啟檔案
for line in jsonFile:               	# 將一筆一筆的 JSON 資料取出
    dt = json.loads(line)           	# 將一筆 JSON 資料轉為 python 字典
    print('話題:', dt['topic'])     	# 取出 key 為 'topic' 的資料
    print('問題:', dt['title'])     	# 取出 key 為 'title' 的資料
    print('回答:', dt['content'])   	# 取出 key 為 'content' 的資料
    print('----'*10)
jsonFile.close()     # 關閉檔案
```
#
```
!pip install opencc-python-reimplemented
```

```
simplified = '卡路里其定义为将1克水在1大气压下提升1所需要的热量。'
print('簡體中文:', simplified)

#s2t === simplified to traditional
cc_s2t = OpenCC('s2t')         			# 建立一個簡體轉繁體的物件
traditional = cc_s2t.convert(simplified)		# 進行轉換
print('繁體中文:', traditional)
```
#
```
import jieba

jieba.set_dictionary('dict.txt.big.txt')	# 設定成繁體字典
# 要斷詞的句子
# sentence = '林夕、方文山、還有五月天的阿信，他們的作詞各有什麼特點？'
sentence = '野生動物園'

# 1. 精準模式
gen = jieba.cut(sentence, cut_all=False)     
print('精準模式: ' + '|'.join(gen) + '\n')	# 為了方便觀察, 我們以 | 將這些字詞串起來

# 2. 全模式
gen = jieba.cut(sentence, cut_all=True)    
print('全模式: ' + '|'.join(gen) + '\n')

# 3. 搜尋引擎模式
gen = jieba.cut_for_search(sentence)   
print('搜尋引擎模式: ' + '|'.join(gen) + '\n') 
```
```
import jieba

jieba.set_dictionary('dict.txt.big.txt')    # 設定成繁體字典
sentence = '林夕、方文山、還有五月天的阿信，他們的作詞各有什麼特點？'

stop_words = set()  # 用來儲存標點符號的 set
with open('標點符號.txt', 'r', encoding='utf-8-sig') as f:
    stop_words = f.read().split('\n')
  #↓斷詞後得到的字詞 List
words = jieba.lcut(sentence, cut_all=False)	# 這次我們改以 lcut() 來斷詞
words = [w for w in words if w not in stop_words] # 1
content = ' '.join(words)	# 以空格重組 List 中的元素
print(content)
```
