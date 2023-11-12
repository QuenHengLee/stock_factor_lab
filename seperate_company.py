from get_data import Data
from database import Database
from backtest import Backtest
from datetime import datetime
import pandas as pd
from finlab_data_frame import CustomDataFrame


class SeperateCompany:
    # 接收SQL下來DB的資料
    def __init__(self):
        # 與資料庫連線
        # 下載全部資料(from database)
        pass

    def backtest_all_quantile(
        self,
        factor_dict,
        quantile=4,
        start_date="2015-01-01",
        end_date="2019-01-01",
        frequency="Q",
    ):
        """
        INPUT: self, 已經被切成N等分的Datafram
        OUTPUT: 存放各個quantile的回測結果的dict
        FUNCTION: 計算各個quantile的回測結果
        """
        all_result_dict = {}
        # 將字串解析成 datetime 物件
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        # 提取日期部分（datetime.date 物件）
        start_date = start_date.date()
        end_date = end_date.date()

        for factor, factor_df in factor_dict.items():
            # start_date = datetime(2015, 1, 1).date()
            # end_date = datetime(2019, 1, 1).date()
            time_period_data = factor_df.loc[start_date:end_date]

            quantile_dict = self.get_quantile_factor(time_period_data, quantile)
            # 创建一个字典，用于存储不同分位数的数据
            result_dict = {}

            for quantile_name, quantile_df in quantile_dict.items():
                # 創建一個Backtest對象
                # 帶入position直為quantile_df
                backtest = Backtest(quantile_df)
                # print("~~quantile_df: ", quantile_df)
                # 将每个分位数的数据存储到字典中
                result_dict[quantile_name] = {
                    "position": backtest.position,
                    "shares_df": backtest.shares_df,
                    "assets": backtest.assets,
                    "stock_data": backtest.stock_data,
                }
                # print("plot of quantile: ", quantile_name)
                # backtest.returns_plot()
            all_result_dict[factor] = result_dict
        # 返回包含不同分位数数据的字典
        return all_result_dict



    def get_quantile_factor(self, factor_df, N=4):
        """
        INPUT: self, 存放單一因子指標的Datafram, 切割成N等分
        OUTPUT: N個DF 每個代表當天每N分位的公司(Quantile 1 的因子值最大)
        FUNCTION: 把所有公司切成N等分
        """

        # 計算每個日期的ROE排名
        rank_df = factor_df.rank(ascending=False, axis=1)

        # 計算每個日期的分位數（根據公司數量和N來定義）
        num_companies = len(factor_df.columns)
        interval = num_companies // N

        # 創建N個DataFrame，用於存放不同分位的公司
        quantile_dfs = [
            ((rank_df > i * interval) & (rank_df <= (i + 1) * interval))
            for i in range(N)
        ]

        # 保持索引和列的一致性
        for df in quantile_dfs:
            df.columns = factor_df.columns
            df.index = factor_df.index

        # 將不同分位的DataFrame存儲在一個字典中，方便根據需要訪問
        quantile_dict = {f"Quantile {i+1}": df for i, df in enumerate(quantile_dfs)}

        # 輸出不同分位的DataFrame
        # for quantile_name, quantile_df in quantile_dict.items():
        #     print(f"{quantile_name}:")
        #     print(quantile_df)
        #     print()

        return quantile_dict


if __name__ == "__main__":
    SeperateCompany = SeperateCompany()
    data = Data()
    factor_df = data.get("report:roe")
    factor_df['ROE']
    a = SeperateCompany.get_quantile_factor(factor_df['ROE'])
