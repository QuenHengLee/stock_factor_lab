## 專案命名規則
- 參考: https://ithelp.ithome.com.tw/articles/10260109
- 類別Class: 使用每個單字的字首用單寫 (如：CapWords)。
- 模組 module: 用小寫及底線 (如：lower_with_under.py)。
- 函數 function: 一律小寫,文字串接的時候使用下底線 (如：def add(): )。
- 全域變數 Globals、 常數 Constants: 一律大寫	(如 PI=3.14)。
- 區域變數 Local variable: 一律小寫,文字串接的時候使用下底線 (val_name = 123)。


## 目前技術上的問題
- Talib 不支支援到最新的python3.11，目前改用python3.8
- Python 的switch架構僅適用於3.10以上
- 兩個項目有點衝突

## 筆記
- https://medium.com/ai%E8%82%A1%E4%BB%94/%E7%94%A8-python-%E5%BF%AB%E9%80%9F%E8%A8%88%E7%AE%97-158-%E7%A8%AE%E6%8A%80%E8%A1%93%E6%8C%87%E6%A8%99-26f9579b8f3a
- Talib指標的公式: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/momentum_indicators.md
