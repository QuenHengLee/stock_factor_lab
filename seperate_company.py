from get_data import Data
from database import Database
from Backtest import Backtest
import pandas as pd


class SeperateCompany:
    # 接收SQL下來DB的資料
    def __init__(self):
        # 與資料庫連線
        # 下載全部資料(from database)
        pass

    """
    INPUT: self, 存放單一因子指標的Datafram, 前N大的公司, 大/小
    OUTPUT: 一個內容為T/F的Dataframe
    FUNCTION: 利用特定切割所有公司(找出前n大)
    """

    def get_ranked_factor(self, factor_df, top_n=10, order="large"):
        # factor_df = self.format_report_data(factor)
        # 初始化一个新的DataFrame，全部设为False
        factor_top_n = pd.DataFrame(
            False, index=factor_df.index, columns=factor_df.columns
        )

        # 对于每个季度，选择factor最高的前10个公司并标记为True
        for quarter in factor_df.index:
            # 計算當天factor值不為nan的公司數量
            factor_values = factor_df.loc[quarter]
            num_companies_with_values = factor_values.count()
            # 如果當天factor不為nan的公司數<top_n --> 全部為false
            if num_companies_with_values < top_n:
                factor_top_n.loc[quarter, :] = False
            else:
                # 判斷要找前N大/小
                if order == "large":
                    top_n_companies = factor_df.loc[quarter].nlargest(top_n).index
                elif order == "small":
                    top_n_companies = factor_df.loc[quarter].nsmallest(top_n).index
                factor_top_n.loc[quarter, top_n_companies] = True

        print(f"Top {top_n} companies ")
        # print(factor_top_n)
        return factor_top_n

    """
    INPUT: self, 存放單一因子指標的Datafram, 切割成N等分
    OUTPUT: N個DF 每個代表當天每N分位的公司(Quantile 1 的因子值最大)
    FUNCTION: 把所有公司切成N等分
    """

    # def backtest_all_quantile(self, quantile_dict):
    #     # 計算每個分位數的報酬率
    #     quantile_returns = pd.DataFrame()
    #     for quantile_name, quantile_df in quantile_dict.items():
    #         print(quantile_name)
    #         # 創建一個Backtest對象
    #         backtest = Backtest(quantile_df)
    #         # 計算每個分位數的報酬率
    #         print(quantile_name, "position\n", backtest.position)
    #         # 持有股票張數
    #         print(quantile_name, "shares_df\n", backtest.shares_df)
    #         # 資產配置(包含: 資產價值、手續費、每次拿來交易的金額、剩餘金額)
    #         print(quantile_name, "assets\n", backtest.assets)
    #         # 這個最重要
    #         # 總表，可以看這個就好
    #         print(quantile_name, "stock_data\n", backtest.stock_data)

    #     # 輸出每個分位數的報酬率
    #     # print(quantile_returns)
    #     return

    def backtest_all_quantile(self, quantile_dict):
        # 创建一个字典，用于存储不同分位数的数据
        result_dict = {}

        for quantile_name, quantile_df in quantile_dict.items():
            # 創建一個Backtest對象
            backtest = Backtest(quantile_df)

            # 将每个分位数的数据存储到字典中
            result_dict[quantile_name] = {
                "position": backtest.position,
                "shares_df": backtest.shares_df,
                "assets": backtest.assets,
                "stock_data": backtest.stock_data,
            }

        # 返回包含不同分位数数据的字典
        return result_dict

    def get_quantile_factor(self, factor_df, N=4):
        # 動態指定分位數的數量
        # N = 4  # 這裡可以設置任意你想要的分位數數量

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
    SeperateCompany.get_ranked_factor(factor_df)
