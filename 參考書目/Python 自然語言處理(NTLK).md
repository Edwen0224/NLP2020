#
```
Python 自然語言處理 Python自然语言处理
雅蘭·薩納卡 (Jalaj Thanaki),機械工業,2018-09-04
```

```
第1章 引言 1 
1.1 自然語言處理 1 
1.2 基礎應用 5 
1.3 高級應用 6 
1.4 NLP和Python相結合的優勢 7 
1.5 nltk環境搭建 7 
1.6 讀者提示 8 
1.7 總結 9 

第2章實踐理解語料庫和資料集 10 
2.1 語料庫 10 
2.2 語料庫的作用 11 
2.3 語料分析 13 
2.4 資料屬性的類型 16 
2.4.1 分類或定性資料屬性 16 
2.4.2 數值或定量資料屬性 17 
2.5 不同檔案格式的語料 18 
2.6 免費語料庫資源 19 
2.7 為NLP應用準備資料集 20 
2.7.1 挑選資料 20 
2.7.2 預處理資料集 20 
2.8 網頁爬取 21 
2.9 總結 23 

第3章 理解句子的結構 24 
3.1 理解NLP的組成 24 
3.1.1 自然語言理解 24 
3.1.2 自然語言生成 25 
3.1.3 NLU和NLG的區別 25 
3.1.4 NLP的分支 26 
3.2 上下文無關文法 26 
3.3 形態分析 28 
3.3.1 形態學 28 
3.3.2 詞素 28 
3.3.3 詞幹 28 
3.3.4 形態分析 28 
3.3.5 詞 29 
3.3.6 詞素的分類 29 
3.3.7 詞幹和詞根的區別 32 
3.4 詞法分析 32 
3.4.1 詞條 33 
3.4.2 詞性標注 33 
3.4.3 匯出詞條的過程 33 
3.4.4詞幹提取和詞形還原的區別 34 
3.4.5 應用 34 
3.5 句法分析 34 
3.6 語義分析 36 
3.6.1 語義分析概念 36 
3.6.2 詞級別的語義 37 
3.6.3 上下位關係和多義詞 37 
3.6.4 語義分析的應用 38 
3.7 消歧 38 
3.7.1 詞法歧義 38 
3.7.2 句法歧義 39 
3.7.3 語義歧義 39 
3.7.4 語用歧義 39 
3.8 篇章整合 40 
3.9 語用分析 40 
3.10 總結 40 

第4章 預處理 42 
4.1 處理原始語料庫文本 42 
4.1.1 獲取原始文本 42 
4.1.2 小寫化轉換 44 
4.1.3 分句 44 
4.1.4 原始文本詞幹提取 46 
4.1.5 原始文本詞形還原 46 
4.1.6 停用詞去除 48 
4.2 處理原始語料庫句子 50 
4.2.1 詞條化 50 
4.2.2 單詞詞形還原 51 
4.3 基礎預處理 52 
4.4 實踐和個性化預處理 57 
4.4.1 由你自己決定 57 
4.4.2 預處理流程 57 
4.4.3 預處理的類型 57 
4.4.4 理解預處理的案例 57 
4.5 總結 62 

第5章 特徵工程和NLP演算法 63 
5.1 理解特徵工程 64 
5.1.1 特徵工程的定義 64 
5.1.2 特徵工程的目的 64 
5.1.3 一些挑戰 65 
5.2 NLP中的基礎特徵 65 
5.2.1 句法分析和句法分析器 65 
5.2.2 詞性標注和詞性標注器 81 
5.2.3 命名實體識別 85 
5.2.4 n元語法 88 
5.2.5 詞袋 89 
5.2.6 語義工具及資源 91 
5.3 NLP中的基礎統計特徵 91 
5.3.1 數學基礎 92 
5.3.2 TF-IDF 96 
5.3.3 向量化 99 
5.3.4 規範化 100 
5.3.5 概率模型 101 
5.3.6 索引 103 
5.3.7 排序 103 
5.4 特徵工程的優點 104 
5.5 特徵工程面臨的挑戰 104 
5.6 總結 104 

第6章高級特徵工程和NLP演算法 106 
6.1 詞嵌入 106 
6.2 word2vec基礎 106 
6.2.1 分佈語義 107 
6.2.2 定義word2vec 108 
6.2.3無監督分佈語義模型中的必需品 108 
6.3 word2vec模型從黑盒到白盒 109 
6.4 基於表示的分佈相似度 110 
6.5 word2vec模型的組成部分 111 
6.5.1 word2vec的輸入 111 
6.5.2 word2vec的輸出 111 
6.5.3word2vec模型的構建模組 111 
6.6 word2vec模型的邏輯 113 
6.6.1 詞彙表構建器 114 
6.6.2 上下文環境構建器 114 
6.6.3 兩層的神經網路 116 
6.6.4 演算法的主要流程 119 
6.7word2vec模型背後的演算法和數學理論 120 
6.7.1word2vec演算法中的基本數學理論 120 
6.7.2詞彙表構建階段用到的技術 121 
6.7.3上下文環境構建過程中使用的技術 122 
6.8 神經網路演算法 123 
6.8.1 基本神經元結構 123 
6.8.2 訓練一個簡單的神經元 124 
6.8.3 單個神經元的應用 126 
6.8.4 多層神經網路 127 
6.8.5 反向傳播演算法 127 
6.8.6word2vec背後的數學理論 128 
6.9生成最終詞向量和概率預測結果的技術 130 
6.10 word2vec相關的一些事情 131 
6.11 word2vec的應用 131 
6.11.1 實現一些簡單例子 132 
6.11.2 word2vec的優勢 133 
6.11.3 word2vec的挑戰 133 
6.11.4在實際應用中使用word2vec 134 
6.11.5 何時使用word2vec 135 
6.11.6 開發一些有意思的東西 135 
6.11.7 練習 138 
6.12 word2vec概念的擴展 138 
6.12.1 para2vec 139 
6.12.2 doc2vec 139 
6.12.3 doc2vec的應用 140 
6.12.4 GloVe 140 
6.12.5 練習 141 
6.13 深度學習中向量化的重要性 141 
6.14 總結 142 

第7章規則式自然語言處理系統 143 
7.1 規則式系統 144 
7.2 規則式系統的目的 146 
7.2.1 為何需要規則式系統 146 
7.2.2 使用規則式系統的應用 147 
7.2.3 練習 147 
7.2.4開發規則式系統需要的資源 147 
7.3 規則式系統的架構 148 
7.3.1從專家系統的角度來看規則式系統的通用架構 149 
7.3.2NLP應用中的規則式系統的實用架構 150 
7.3.3NLP應用中的規則式系統的定制架構 152 
7.3.4 練習 155 
7.3.5 Apache UIMA架構 155
```

```


```
