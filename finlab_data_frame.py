import pandas as pd
import numpy as np
from datetime import datetime

# 參考來源
# finlab官方1: https://doc.finlab.tw/reference/dataframe/
# finlab官方2: https://doc.finlab.tw/details/backtest_decorator/
class CustomDataFrame(pd.DataFrame):
    def __and__(self, other):
        # 取 df1、df2 的索引聯集和列交集
        common_columns = self.columns.intersection(other.columns)
        common_index = self.index.union(other.index)

        # 使用前向填充方法填充缺失值
        self = self.reindex(common_index).fillna(method='ffill', inplace=False)
        other = other.reindex(common_index).fillna(method='ffill', inplace=False)

        # 創建新的 CustomDataFrame，並初始化為 False
        result = CustomDataFrame(False, columns=common_columns, index=common_index)

        # 對 df1 和 df2 中的相同列和索引進行邏輯 AND 運算
        result[common_columns] = self[common_columns] & other[common_columns]

        return result





# -----------------------------------------------------------
# 創建兩個示例 CustomDataFrame，使用日期作為索引
data1 = {'A': [False, True ], 'B': [False, True]}
data2 = {'A': [True, False, True], 'C': [False, True, True]}
date1 = [datetime(2023, 1, 1) , datetime(2023, 1, 3)]
date2 = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]

df1 = CustomDataFrame(data1, index=date1)
df2 = CustomDataFrame(data2, index=date2)

# 使用自定義的 AND 運算子方法
result = df1 & df2

print("AND 運算的結果:")
print(result)
