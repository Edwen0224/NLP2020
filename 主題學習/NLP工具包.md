#
```

```
```
https://thoughts.blueplanet.com.tw/home
```

```
2. stanford parser
http://nlp.stanford.edu/software/lex-parser.shtml 
http://nlp.stanford.edu/software/segmenter.shtml
http://nlp.stanford.edu/software/tagger.shtml
‪#簡體‬ #斷詞 #詞性標記 #句型結構 #修飾關係 ‪#NER‬
1. 處理繁體建議先轉成簡體以得到較佳效果
2. 可下載單機版，可自己訓練繁體模型（不知道有沒有人分享出來）
3. 支援多種程式語言：JAVA, Python, Ruby, PHP
4. 詞性有十幾種
5. 有NER 具名實體辨識

3. mmseg 斷詞
http://technology.chtsai.org/mmseg/ 
#繁體 #斷詞 ‪#快‬
可下載單機版，可自己訓練繁體模型，可使用自訂字典
我執行的時候跳出視窗說windows版本不符
4.SCWS 中文分词
http://www.xunsearch.com/scws/
雖然是中國開發者做的，但試過處理正體中文也 OK ，只是詞庫並不是很豐富就是了。詞庫可以擴充，主要針對 PHP 開發者。

5.NLTK
python的自然語言處理包，需要先斷詞
http://www.nltk.org/book/

6.CNLP
師大語言所製作的中文處理整合包(基於NLTK)，根據網頁說明，能處理經中研院斷詞、詞性標記過的文本，其他系統處理的斷詞不曉得能不能適用
http://tm.itc.ntnu.edu.tw/CNLP/?q=node/5

7.結巴中文分詞（簡中）
https://github.com/fxsjy/jieba

8. FudanNLP（簡中）
https://github.com/xpqiu/fnlp/

9. Glove
Create word embeddings for further analysis
http://nlp.stanford.edu/projects/glove/

10. OpenCC
繁簡轉換
https://github.com/BYVoid/OpenCC

11. ansj
簡體斷詞
http://www.nlpcn.org/demo
https://github.com/NLPchina/ansj_seg

12. 國教院分詞系統
中研院 CKIP 的衍生系統，據國教院的同仁說，新近詞的收量較大，跑起來也稍快些。
http://120.127.233.228/Segmentor/
另外還附有一個語料索引系統：http://120.127.233.228/Concordancer/

13. cjknife
ref: http://logbot.g0v.tw/channel/g0v.tw/2015-03-26#94

14. Unicode Normalization
主要是用在清理一些看起來長的一樣但實際字碼不同的字
官方定義： http://unicode.org/reports/tr15/
PHP: http://php.net/manual/en/class.normalizer.php
JS: https://github.com/walling/unorm

15.JIEBA 結巴中文斷詞
介紹簡報：https://speakerdeck.com/fukuball/jieba-jie-ba-zhong-wen-duan-ci

16.Articut 中文斷詞暨語意詞性標記系統
商用等級的，無需自己準備資料做機器學習或模型訓練，可自定字典，也隨時可提出修正需求給原廠。300 元可處理 10 萬字。
斷詞同時也做好了中文人名偵測、代名詞推理、語意詞性標記的推理…等。
介紹簡報：https://ppt.cc/fYCnOx
試用網站：https://api.droidtown.co 
Github API 專案：https://github.com/Droidtown/ArticutAPI
FB：https://www.facebook.com/Articut   
```
# 中文處理工具
```
中文處理工具簡介
https://g0v.hackmd.io/fR51fyEcQYOVIGSCanO3TA
https://twilightzone.gitlab.io/note/CS/NLP.html


1. 中研院CKIP parser
http://ckipsvr.iis.sinica.edu.tw/
http://parser.iis.sinica.edu.tw/
‪#繁體‬ ‪#斷詞‬ ‪#詞性標記‬ ‪#句型結構‬ ‪#修飾關係‬
1. 有點慢，準確率最高
2. 可透過web service呼叫（詞性較粗）或爬網頁（詞性較細）。
3. 可細分四十多種詞性，如名詞可細分為地方名詞、普通名詞，專有名詞等。

千呼萬喚十多年！中研院終於開源釋出國產自動化中文斷詞工具，正式採用GPL 3.0釋出
中研院近日正式開源釋出了自行研發多年的中文斷詞程式，提供給在臺灣從事中文自然語言處理研究的學術圈或開發者來使用，目前已放上GitHub平臺。
文/余至浩 | 2019-09-04發表
https://www.ithome.com.tw/news/132838
https://github.com/ckiplab/ckiptagger
https://ckip.iis.sinica.edu.tw/

结巴中文分词
https://github.com/fxsjy/jieba

A modified version of Jieba
https://github.com/amigcamel/Jseg
```
```
NLU (Natural Language understanding) & NLG (Natural Language Generation)
NLU ( Lang to Structure data ) : 查詢、詐騙偵測、情緒分析..
NLG ( Structure data to Lang ) : 新聞寫作
NLU + NLG : 機器翻譯、聊天機器人
```
```
自動學習特徵 (domain knowledge)
在傳統 Machine Learning 上我們必須要給予一些 feature 來輸入 model 進行任務的分析，
但在 Deep Learning 上我們省略了 Feature Extraction 的步驟，機器可以自行學習出 features。
在論文 “Convolutional Neural Network for sentence classification“ 利用 CNN model 來進行 end2end 特徵的萃取及分類。


Pre-training + training ( unsupervised + supervised )
在 NLP 中，我們希望要先對 text 進行 embedding (pre-training) ，給予一個比較好的初始值，再丟進 model 進行訓練 ( training )，可以節省計算成本。
BERT : pre-training (embedding/model) + training (finetune)
```
```
Q : 中文的 NLP 與英文在處理上有什麼樣子的差異 ?
A : 中英文的差異除斷詞問題、中文的語法結構較英文鬆散靈活、中文上主詞會很常省略。因此，在中文長篇上會較英文難上許多。
此外，英文的量詞發達、中文太常使用倒裝句，這些都會造就 NLP 上的處理難度。

ttps://allen108108.github.io/blog/2019/11/01/中文自然語言處理 (NLP) 的進展與挑戰/
```
# 中研院詞庫小組(CKIP)
```
中研院跨所的一個中文計算語言研究小組，五個主要研究方向：深度學習、知識表達、自然語言理解、知識擷取、聊天機器人

CkipTagger
```
```
中文詞知識庫小組
Chinese Knowledge and Information Processing (CKIP)
馬偉雲 助研究員
研究專長: 自然語言處理、自然語言理解、機器翻譯

計畫:
A. 具推薦功能的聊天機器人: 新聞聊天機器人，與Line合作。
B. 不限主題的閒聊機器人:

專有名詞辨識 / 實體辨識 (Named Entity Recognition, NER):
A. 語料蒐集。
B. 中文字詞轉成向量表達(word2vec)。
C. CKIP 中文斷詞系統和中文剖析系統擷取語法語義特徵。
D. 深度遞迴類神經網路模型，預測實體位置語類別。

指代消解 (Coreference Resolution):
A. 中文分詞程式中，多半討論分詞歧義的問題，較少討論unseen 詞彙的問題。如何解決unseen詞彙的分詞問題呢? 一般以高頻出現的關鍵字作為分詞依據。
B. 指式代名詞會使專有名詞的出現頻率降低，因而造成誤判的狀況。
C. 透過指代消解的處理，可以將被替換過的字詞還原成原有的意思，以提高權重計算的次數，增加檢索的正確性。

輿情分析系統
聊天機器人

https://medium.com/@chiangyulun0914/%E4%B8%AD%E7%A0%94%E9%99%A2%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8%E7%A0%94%E7%A9%B6%E6%89%80-884a778fd4af
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

