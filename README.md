## 使用Anaconda
- 建立一個自己的環境
- 並匯入requirements.txt 的套件

## 目前技術上的問題
- Talib 不支支援到最新的python3.11，目前改用python3.8
- 但目前沒用到Talib
- Python 的switch架構僅適用於3.10以上
- 兩個項目有點衝突

## 筆記
- https://medium.com/ai%E8%82%A1%E4%BB%94/%E7%94%A8-python-%E5%BF%AB%E9%80%9F%E8%A8%88%E7%AE%97-158-%E7%A8%AE%E6%8A%80%E8%A1%93%E6%8C%87%E6%A8%99-26f9579b8f3a
- Talib指標的公式: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/func_groups/momentum_indicators.md

## 可能會用到基本語法
主要分成以下四種:
|            |存取raw data|dataframe操作|回測        |顯示report   |
|:------------:|------------|------------|------------|------------|
|功能描述|從資料庫中取得資料，包含股價(開、高、低、收)以及財務報表，並儲存成dataframe形式|針對儲存成dataframe形式之資料做處理|回測模擬股票部位所產生的淨值報酬率|策略回測基礎報告|
|語法|1. 建立DB連線<br> data=Data()<br>2. 取得資料<br>data.get("你想取得的資料")|1. 取得移動平均<br>df.average(n_windows)<br>2. 買入訊號持續為True直到出場條件<br>df.hold_until(exits)<br>3. 根據因子要切割成幾群<br>df.divide_slice(quantile)|1. 回測一個position，並回傳一個report物件<br>report=backtest.sim(position, resample)<br>2. 回測多個position，並回傳一個report的dict結構<br>report_conditions=sim_conditions(conditions, resample)<br>註:conditoins是選股條件的集合，應為dict結構|1. 顯示單一position圖組<br>report.display()<br>2. 顯示CAGR、MDD、sharpe ratio等數據<br>report.get_status()<br>3. 顯示report_conditions累計回報比較<br>report_collection.plot_creturns().show()<br>4. 顯示report_conditions各種數據比較，有柱狀圖、熱力圖<br>柱狀圖:report_collection.plot_stats('bar').show()<br>熱力圖:report_collection.plot_stats('heatmap')|
