from get_data import Data
from database import Database
from backtest import *
from datetime import datetime
import pandas as pd
from dataframe import CustomDataFrame
import pandas as pd
import numpy as np

def cal_interpolated_of_df(df):
    '''
    INPUTS:
        df: 想要計算內插值的資料dataframe
    RETURN:
        interpolated_df: 計算完因子內插值後的dataframe
    FUNCTION:
        以ROW為基準，計算每一ROW的內插值，最大1最小0
    '''
    # 計算每行的最大值和最小值
    max_values = df.max(axis=1)
    min_values = df.min(axis=1)

    # 計算內插值
    interpolated_df = (df.sub(min_values, axis=0)).div((max_values - min_values), axis=0)
    
    return interpolated_df

def cal_factor_sum_df_interpolated(factor_df_dict, factor_ratio_dict,quantile=4):
    '''
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        factor_ratio_dict: 一個字典，包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        factor_sum_df_interpolated: 雙因子內插值相加後的加權總分
    FUNCTION:
        該因子選股的方法是根據台股研究室的內插法
        計算多個因子內插值的加權總分，如果有任一因子為nan，其他因子不為nan，則加總也是nan
        最後根據因子切割的大小quantile，回傳該權重的position
    '''
    # 確保輸入的因子數量和比重數量相等
    if len(factor_df_dict) != len(factor_ratio_dict):
        raise ValueError("因子數量和比重數量不相等")

    # 計算因子DF的內插值
    factor_df_interpolated = {name: cal_interpolated_of_df(df) for name, df in factor_df_dict.items()}

    # 將每個因子的內插值乘上對應的比重
    factor_interpolated_weighted = {name: interpolated * factor_ratio_dict[name] for name, interpolated in factor_df_interpolated.items()}

    # 將所有因子的加權內插值相加，得加權總分
    # 並轉成CustomDataFrame
    factor_sum_df_interpolated = sum(factor_interpolated_weighted.values())
    factor_sum_df_interpolated = CustomDataFrame(factor_sum_df_interpolated)

    # 回傳多因子權重加總後的dataframe
    return factor_sum_df_interpolated.divide_slice(quantile)


def factor_analysis_multi_ratio(factor_df_dict, quantile=4):
    '''
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        factor_sum_df_interpolated_dict: 各種因子權重組合的內插值相加後的加權總分，回傳一個包含多個df的dict
    FUNCTION:
        該因子選股的方法是根據台股研究室的內插法
        計算多個因子內插值的加權總分，如果有任一因子為nan，其他因子不為nan，則加總也是nan
        計算各種因子權重比例的組合，提供不同的positon
    '''
    # 回測以下組組合的
    muiti_ratio_dict = {
        {'factor1': 0,'factor2': 1,},
        {'factor1': 0.1,'factor2': 0.9,},
        {'factor1': 0.2,'factor2': 0.8,},
        {'factor1': 0.3,'factor2': 0.7,},
        {'factor1': 0.4,'factor2': 0.6,},
        {'factor1': 0.5,'factor2': 0.5,},
        {'factor1': 0.6,'factor2': 0.4,},
        {'factor1': 0.7,'factor2': 0.3,},
        {'factor1': 0.8,'factor2': 0.2,},
        {'factor1': 0.9,'factor2': 0.1,},
        {'factor1': 1,'factor2': 0,}
    }

    

    # 每種權重組合都帶入計算
    for ratio in muiti_ratio_dict:
        cal_factor_sum_df_interpolated(factor_df_dict, ratio, quantile)

    # TODO...

def factor_analysis_two_factor(factor_df_dict, quantile=4):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        quantile: 打算將因子切割成幾等分       
    RETURN:
        各分位的position，回傳一個包含多個df的dict
    FUNCTION:
        實現Achieving Alpha的雙因子選股方法，強調第一個因子，弱化第二個因子
        
    """
    # 從Input擷取個因子的DF
    factor_1_df = factor_df_dict[0]
    factor_2_df = factor_df_dict[1]
    # 先將第一個因子根據quantile值做切割
    factor_1_slice = 








if __name__ == "__main__":

    # # 生成 10x10 的隨機數據，其中大約 20% 的元素為 NaN
    # np.random.seed(42)
    # data = np.random.randint(0, 100, size=(10, 10)).astype(float)  # 將數據類型設為浮點型
    # mask = np.random.choice([True, False], size=data.shape, p=[0.2, 0.8])
    # data[mask] = np.nan
    # # 將數據轉換成 DataFrame
    # factor_1 = pd.DataFrame(data, columns=[f'Col{i+1}' for i in range(10)], index=[f'Row{j+1}' for j in range(10)])

    # # 生成 10x10 的隨機數據，其中大約 20% 的元素為 NaN
    # np.random.seed(98)
    # data = np.random.randint(0, 100, size=(10, 10)).astype(float)  # 將數據類型設為浮點型
    # mask = np.random.choice([True, False], size=data.shape, p=[0.2, 0.8])
    # data[mask] = np.nan
    # # 將數據轉換成 DataFrame
    # factor_2 = pd.DataFrame(data, columns=[f'Col{i+1}' for i in range(10)], index=[f'Row{j+1}' for j in range(10)])
   
    # # 呼叫因子分析
    # result = cal_factor_sum_df_interpolated(factor_1, factor_2, 0.7,0.3)

    # 示例使用方式
    factor_df_dict = {
        'factor1': pd.DataFrame(np.random.rand(5, 5)),
        'factor2': pd.DataFrame(np.random.rand(5, 5)),
        # 可以加入其他因子
    }

    factor_ratio_dict = {
        'factor1': 0.5,
        'factor2': 0.5,
        # 可以加入其他比重
    }

    a,b,c,d = cal_factor_sum_df_interpolated(factor_df_dict, factor_ratio_dict)

    result = cal_factor_sum_df_interpolated(factor_df_dict, factor_ratio_dict)
    
        







# class SeperateCompany:
#     # 接收SQL下來DB的資料
#     def __init__(self):
#         # 與資料庫連線
#         # 下載全部資料(from database)
#         pass

#     # def backtest_all_quantile(
#     #     self,
#     #     factor_dict,
#     #     quantile=4,
#     #     start_date="2015-01-01",
#     #     end_date="2019-01-01",
#     #     frequency="Q",
#     # ):
#     #     """
#     #     INPUT: self, 已經被切成N等分的Datafram
#     #     OUTPUT: 存放各個quantile的回測結果的dict
#     #     FUNCTION: 計算各個quantile的回測結果
#     #     """
#     #     all_result_dict = {}
#     #     # 將字串解析成 datetime 物件
#     #     start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     #     end_date = datetime.strptime(end_date, "%Y-%m-%d")
#     #     # 提取日期部分（datetime.date 物件）
#     #     start_date = start_date.date()
#     #     end_date = end_date.date()

#     #     for factor, factor_df in factor_dict.items():
#     #         # start_date = datetime(2015, 1, 1).date()
#     #         # end_date = datetime(2019, 1, 1).date()
#     #         time_period_data = factor_df.loc[start_date:end_date]

#     #         quantile_dict = self.get_quantile_factor(time_period_data, quantile)
#     #         # 创建一个字典，用于存储不同分位数的数据
#     #         result_dict = {}

#     #         for quantile_name, quantile_df in quantile_dict.items():
#     #             # 創建一個Backtest對象
#     #             # 帶入position直為quantile_df
#     #             backtest = Backtest(quantile_df)
#     #             # print("~~quantile_df: ", quantile_df)
#     #             # 将每个分位数的数据存储到字典中
#     #             result_dict[quantile_name] = {
#     #                 "position": backtest.position,
#     #                 "shares_df": backtest.shares_df,
#     #                 "assets": backtest.assets,
#     #                 "stock_data": backtest.stock_data,
#     #             }
#     #             # print("plot of quantile: ", quantile_name)
#     #             # backtest.returns_plot()
#     #         all_result_dict[factor] = result_dict
#     #     # 返回包含不同分位数数据的字典
#     #     return all_result_dict

#     def get_quantile_factor(self, factor_df, N=4):
#         """
#         INPUT: self, 存放單一因子指標的Dataframe, 切割成N等分
#         OUTPUT: N個DF 每個代表當天每N分位的公司(Quantile 1 的因子值最大)
#         FUNCTION: 把所有公司切成N等分
#         """

#         # 計算每個日期的ROE排名
#         rank_df = factor_df.rank(ascending=False, axis=1)

#         # 計算每個日期的分位數（根據公司數量和N來定義）
#         num_companies = len(factor_df.columns)
#         interval = num_companies // N

#         # 創建N個DataFrame，用於存放不同分位的公司
#         quantile_dfs = [
#             ((rank_df > i * interval) & (rank_df <= (i + 1) * interval))
#             for i in range(N)
#         ]

#         # 保持索引和列的一致性
#         for df in quantile_dfs:
#             df.columns = factor_df.columns
#             df.index = factor_df.index

#         # 使用tuple來存放不同分位的DataFrame
#         quantile_tuple = tuple(
#             (f"Quantile {i+1}", df) for i, df in enumerate(quantile_dfs)
#         )

#         # 輸出不同分位的DataFrame
#         # for quantile_name, quantile_df in quantile_tuple:
#         #     print(f"{quantile_name}:")
#         #     print(quantile_df)
#         #     print()

#         return quantile_tuple


# if __name__ == "__main__":
#     SeperateCompany = SeperateCompany()
#     data = Data()
#     factor_df = data.get("report:roe")
#     factor_df["ROE"]
#     a, b, c, d = SeperateCompany.get_quantile_factor(factor_df["ROE"])
