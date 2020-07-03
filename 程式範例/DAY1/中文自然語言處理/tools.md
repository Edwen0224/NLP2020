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
