#
```
TensorFlow 自然語言處理｜善用 Python 深度學習函式庫，教機器學會自然語言 (Natural Language Processing with TensorFlow)
Thushan Ganegedara 著 藍子軒 譯
碁峰資訊,2019-09-09
```

```

```

```
章　自然語言處理簡介 1
1.1　什麼是自然語言處理 1
1.2　自然語言處理的任務 2
1.3　傳統的自然語言處理方法 3
1.3.1　理解傳統方法 4
1.3.2　傳統方法的缺點 7
1.4　自然語言處理的深度學習方法? 8
1.4.1　深度學習的歷史 8
1.4.2　深度學習和NLP的當前狀況 9
1.4.3　理解一個簡單的深層模型—全連接神經網路
1.5　本章之外的學習路線 12
1.6　技術工具簡介 14
1.6.1　工具說明 15
1.6.2　安裝Python和scikit-learn 15
1.6.3　安裝Jupyter Notebook 15
1.6.4　安裝TensorFlow 16
1.7　總結17

第2章　理解TensorFlow 18
2.1　TensorFlow是什麼18
2.1.1　TensorFlow入門19
2.1.2　TensorFlow客戶端詳細介紹 21
2.1.3　TensorFlow架構：當你執行用戶端時發生了什麼 21
2.1.4　Cafe Le TensorFlow：使用類比理解TensorFlow 23
2.2　輸入、變數、輸出和操作 24
2.2.1　在TensorFlow中定義輸入 25
2.2.2　在TensorFlow中定義變數 30
2.2.3　定義TensorFlow輸出31
2.2.4　定義TensorFlow操作31
2.3　使用作用域重用變數 40
2.4　實現個神經網路 42
2.4.1　準備數據 43
2.4.2　定義TensorFLow圖43
2.4.3　運行神經網路 45
2.5　總結46

第3章　Word2vec——學習詞嵌入 48
3.1　單詞的表示或含義是什麼 49
3.2　學習單詞表示的經典方法 49
3.2.1　WordNet—使用外部詞彙知識庫來學習單詞表示 50
3.2.2　獨熱編碼表示方式 53
3.2.3　TF-IDF方法53
3.2.4　共現矩陣 54
3.3　Word2vec—基於神經網路學習單詞表示 55
3.3.1　練習：queen = king – he she嗎56
3.3.2　為學習詞嵌入定義損失函數 58
3.4　skip-gram演算法59
3.4.1　從原始文本到結構化的資料 59
3.4.2　使用神經網路學習詞嵌入 60
3.4.3　使用TensorFlow實現skip-gram 67
3.5　連續詞袋演算法 69
3.6　總結71

第4章　Word2vec 72
4.1　原始skip-gram演算法72
4.1.1　實現原始skip-gram演算法73
4.1.2　比較原始skip-gram演算法和改進的skip-gram演算法75
4.2　比較skip-gram演算法和CBOW演算法75
4.2.1　性能比較 77
4.2.2　哪個更勝一籌：skip-gram還是CBOW 79
4.3　詞嵌入演算法的擴展 81
4.3.1　使用unigram分佈進行負採樣 81
4.3.2　實現基於unigram的負採樣81
4.3.3　降採樣：從概率上忽視常用詞 83
4.3.4　實現降採樣 84
4.3.5　比較CBOW及其擴展演算法 84
4.4　近的skip-gram和CBOW的擴展演算法 85
4.4.1　skip-gram演算法的限制 85
4.4.2　結構化skip-gram演算法85
4.4.3　損失函數 86
4.4.4　連續視窗模型 87
4.5　GloVe：全域向量表示 88
4.5.1　理解GloVe 88
4.5.2　實現GloVe 89
4.6　使用Word2vec進行文檔分類 90
4.6.1　資料集91
4.6.2　用詞向量進行文檔分類 91
4.6.3　實現：學習詞嵌入 92
4.6.4　實現：詞嵌入到文檔嵌入 92
4.6.5　文本聚類以及用t-SNE視覺化文檔嵌入 93
4.6.6　查看一些特異點 94
4.6.7　實現：用K-means對文檔進行分類/聚類95
4.7　總結96

第5章　用卷積神經網路進行句子分類 97
5.1　介紹卷積神經網路 97
5.1.1　CNN基礎 97
5.1.2　卷積神經網路的力量 0
5.2　理解卷積神經網路 0
5.2.1　卷積操作 0
5.2.2　池化操作 3
5.2.3　全連接層 4
5.2.4　組合成完整的CNN 5
5.3　練習：在MNIST資料集上用CNN進行圖片分類 5
5.3.1　關於資料 6
5.3.2　實現CNN 6
5.3.3　分析CNN產生的預測結果 8
5.4　用CNN進行句子分類 9
5.4.1　CNN結構 1
5.4.2　隨時間池化 112
5.4.3　實現：用CNN進行句子分類 112
5.5　總結115

第6章　遞迴神經網路 116
6.1　理解遞迴神經網路 116
6.1.1　前饋神經網路的問題 117
6.1.2　用遞迴神經網路進行建模 118
6.1.3　遞迴神經網路的技術描述 119
6.2　基於時間的反向傳播 119
6.2.1　反向傳播的工作原理 120
6.2.2　為什麼RNN不能直接使用反向傳播 120
6.2.3　基於時間的反向傳播：訓練RNN 121
6.2.4　截斷的BPTT：更有效地訓練RNN 121
6.2.5　BPTT的限制：梯度消失和梯度 122

6.3　RNN的應用 123
6.3.1　一對一RNN 123
6.3.2　一對多RNN 123
6.3.3　多對一RNN 124
6.3.4　多對多RNN 124
6.4　用RNN產生文本 125
6.4.1　定義超參數 125
6.4.2　將輸入隨時間展開用於截斷的BPTT 125
6.4.3　定義驗證資料集 126
6.4.4　定義權重和偏置 126
6.4.5　定義狀態持續變數 127
6.4.6　用展開的輸入計算隱藏狀態和輸出 127
6.4.7　計算損失 128
6.4.8　在新文本片段的開頭重置狀態 128
6.4.9　計算驗證輸出 128
6.4.　計算梯度和優化 129
6.4.11　輸出新生成的文字區塊 129
6.5　評估RNN的文本結果輸出 130
6.6　困惑度：衡量文本結果的品質 131
6.7　有上下文特徵的遞迴神經網路：更長記憶的RNN 132
6.7.1　RNN-CF的技術描述 132
6.7.2　實現RNN-CF 133
6.7.3　RNN-CF產生的文本 138
6.8　總結140

第7章　長短期記憶網路 142
7.1　理解長短期記憶網路 142
7.1.1　LSTM是什麼 143
7.1.2　更詳細的LSTM 144
7.1.3　LSTM與標準RNN的區別 149
7.2　LSTM如何解決梯度消失問題 150
7.2.1　改進LSTM 152
7.2.2　貪婪採樣 153
7.2.3　集束搜索 153
7.2.4　使用詞向量 154
7.2.5　雙向LSTM（BiLSTM）155
7.3　其他LSTM的變體156
7.3.1　窺孔連接 156
7.3.2　門迴圈單元 157
7.4　總結159

第8章　LSTM應用：文本生成 160
8.1　資料集160
8.1.1　關於資料集 160
8.1.2　數據預處理 162
8.2　實現LSTM 162
8.2.1　定義超參數 163
8.2.2　定義參數 163
8.2.3　定義LSTM單元及操作 165
8.2.4　定義輸入和標籤 165
8.2.5　定義處理序列資料所需的序列計算 166
8.2.6　定義優化器 167
8.2.7　隨時間衰學習率 167
8.2.8　做預測168
8.2.9　計算困惑度（損失） 168
8.2.　重置狀態 169
8.2.11　貪婪採樣避免單峰 169
8.2.12　生成新文本 169
8.2.13　生成的文本樣例 170
8.3　LSTM與窺孔LSTM和GRU對比171
8.3.1　標準LSTM 171
8.3.2　門控迴圈單元（GRU）172
8.3.3　窺孔LSTM 174
8.3.4　訓練和驗證隨時間的困惑度 175
8.4　改進LSTM：集束搜索 176
8.4.1　實現集束搜索 177
8.4.2　集束搜索生成文本的示例 179
8.5　LSTM改進：用單詞替代n-gram生成文本 179
8.5.1　維度災難 179
8.5.2　Word2vec補救180
8.5.3　使用Word2vec生成文本180
8.5.4　使用LSTM-Word2vec和集束搜索生成的文本示例 181
8.5.5　隨時間困惑度 182
8.6　使用TensorFlow RNN API 183
8.7　總結186

第9章　LSTM應用：圖像標題生成 188
9.1　瞭解資料 188
9.1.1　ILSVRC ImageNet資料集189
9.1.2　MS-COCO資料集189
9.2　圖像標題生成實現路徑 191
9.3　使用CNN提取圖像特徵 193
9.4　實現：使用VGG-16載入權重和推理 193
9.4.1　構建和更新變數 194
9.4.2　預處理輸入 195
9.4.3　VGG-16推斷196
9.4.4　提取圖像的向量化表達 197
9.4.5　使用VGG-16預測類別概率 197
9.5　學習詞嵌入 198
9.6　準備輸入LSTM的標題198
9.7　生成LSTM的資料199
9.8　定義LSTM 201
9.9　定量評估結果 203
9.9.1　BLEU 203
9.9.2　ROUGE 204
9.9.3　METEOR 204
9.9.4　CIDEr 206
9.9.5　模型隨著時間變化的BLEU-4 206
9.　為測試圖像生成標題 207
9.11　使用TensorFlow RNN API和預訓練的GloVe詞向量2
9.11.1　載入GloVe詞向量2
9.11.2　清洗數據 212
9.11.3　使用TensorFlow RNN API和預訓練的詞嵌入 213
9.12　總結218

第章　序列到序列學習：神經機器翻譯 220
.1　機器翻譯 220
.2　機器翻譯簡史 221
.2.1　基於規則的翻譯 221
.2.2　統計機器翻譯（SMT）222
.2.3　神經機器翻譯（NMT）223
.3　理解神經機器翻譯 225
.3.1　NMT原理 225
.3.2　NMT架構 226
.4　為NMT系統準備資料 228
.4.1　訓練階段 229
.4.2　反轉源句 229
.4.3　測試階段 230
.5　訓練NMT 230
.6　NMT推理 231
.7　BLEU評分：評估機器翻譯系統 232
.7.1　修正的度 232
.7.2　簡短懲罰項 233
.7.3　終BLEU得分233
.8　從頭開始實現NMT：德語到英語的翻譯 233
.8.1　資料介紹 234
.8.2　處理資料 234
.8.3　學習詞嵌入 235
.8.4　定義編碼器和解碼器 236
.8.5　定義端到端輸出計算 238
.8.6　翻譯結果 239
.9　結合詞嵌入訓練NMT 241
.9.1　化資料集詞彙表和預訓練詞嵌入之間的匹配 241
.9.2　將嵌入層定義為TensorFlow變數243
.　改進NMT 245
..1　教師強迫 246
..2　深度LSTM 247
.11　注意力247
.11.1　突破上下文向量瓶頸 247
.11.2　注意力機制細節 248
.11.3　注意力NMT的翻譯結果 253
.11.4　源句子和目標句子注意力視覺化 254
.12　序列到序列模型的其他應用：聊天機器人 256
.12.1　訓練聊天機器人 256
.12.2　評估聊天機器人：圖靈測試 257
.13　總結258

1章　自然語言處理的現狀與未來 259
11.1　NLP現狀 259
11.1.1　詞嵌入260
11.1.2　神經機器翻譯 264
11.2　其他領域的滲透 266
11.2.1　NLP與電腦視覺結合 266
11.2.2　強化學習 268
11.2.3　NLP生成式對抗網路 269

11.3　走向通用人工智慧 270
11.3.1　一個模型學習 271
11.3.2　聯合多工模型：為多個NLP任務生成神經網路 272

11.4　媒體NLP 273
11.4.1　媒體中的謠言檢測 274
11.4.2　媒體中的情緒檢測 274
11.4.3　分析推特中的政治框架 274

11.5　湧現的新任務 275
11.5.1　諷刺檢測 275
11.5.2　語言基礎 276
11.5.3　使用LSTM略讀文本276
11.6　新興的機器學習模型 277
11.6.1　階段LSTM 277
11.6.2　擴張RNN（DRNN） 278
11.7　總結278
11.8　參考文獻 279

附錄　數學基礎與TensorFlow 282


```
