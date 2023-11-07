from database import Database
from format_data import *
import talib
import pandas as pd
import numpy as np

# Abstract API：
from talib import abstract


class Data:
    # 物件初始化: 接收SQL下來DB的資料
    def __init__(self):
        # 與資料庫連線 & 下載全部資料(from database)
        # 根據config.ini 建立DB連線
        self.db = Database()
        # 初始化stock price 資料
        self.raw_price_data = self.db.get_daily_stock()
        self.all_price_dict = self.handle_price_data()
        # 初始化stock price 資料
        self.raw_report_data = self.db.get_finance_report()

    # 從format_data取得處理後得資料
    def format_price_data(self, item):
        return format_price_data(self.raw_price_data, item)

    def format_report_data(self, factor):
        return format_report_data(self.raw_report_data, factor)

    def handle_price_data(self):
        return handle_price_data(self.raw_price_data)

    """
    INPUT: self, dataset(str)帶入想要的資料名稱Ex. price:close、report:roe
    OUTPUT: 一個內容為所有公司股價/財報相關資料的Dataframe
    FUNCTION: 從DB中取得資料
    """

    # 取得資料起點
    def get(self, dataset):
        # 使用 lower() 方法將字串轉換為小寫
        dataset = dataset.lower()
        # 使用 split() 函數按 ":" 分隔字串
        parts = dataset.split(":")
        if len(parts) == 2:
            subject = parts[0]
            item = parts[1]
        else:
            print("輸入格式錯誤(Ex. price:close)")

        if subject == "price":
            # 呼叫處理開高低收的FUNCT
            price_data = self.all_price_dict[item]
            return price_data
        elif subject == "report":
            # 財報資料的Header為大寫
            item = item.upper().replace(" ", "")
            # 可能會有多個財報資料，以逗號加空格為分隔符
            # 將字串以逗號加空格為分隔符，分割成元素列表
            elements = item.split(",")
            # 創建一個以元素為鍵，空字串為值的字典
            element_dict = {element: "" for element in elements}
            # 使用迴圈遍歷字典的鍵值對
            for key, value in element_dict.items():
                # 呼叫處理財報的FUNCT
                element_dict[key] = self.format_report_data(key)

            # 有多個財報資料時，回傳一個字典
            return element_dict
        else:
            print("目前資料來源有price、report")

    def indicator(
        self, indname, adjust_price=False, resample="D", market="TW_STOCK", **kwargs
    ):
        # 先取得所有公司list，因為計算指標是一間一間算
        all_company_symbol = get_all_company_symbol(data.raw_price_data)

        # 先計算該指標會回傳幾個值
        tmp_company_daily_price = get_each_company_daily_price(
            self.raw_price_data, all_company_symbol[0]
        )
        num_of_return = get_number_of_indicator_return(indname, tmp_company_daily_price)

        # 再根據回傳值的數量動態宣告N個dataframe在一個tuple中
        empty_dataframe = pd.DataFrame()
        dataframe_tuple = tuple()
        # 複製空的df到tuple中
        for _ in range(num_of_return):
            df = empty_dataframe.copy()
            dataframe_tuple += (df,)

        # 用巢狀迴圈逐一填入N公司的M個回傳指標資料
        for i in range(num_of_return):
            for company_symbol in all_company_symbol:
                df = get_each_company_daily_price(self.raw_price_data, company_symbol)
                # df 為存放單一公司所有日期的開高低收量資料(col小寫)
                result = eval("abstract." + indname + "(df)")
                # 假如只有回傳一個值，回以series呈現，這邊要轉成dataframe
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                else:
                    pass  # df不用再轉df
                dataframe_tuple[i][company_symbol] = result.iloc[:, i]

        if num_of_return == 1:
            return dataframe_tuple[0]
        else:
            return dataframe_tuple
        # result_df = pd.DataFrame()
        # for company_symbol in all_company_symbol:
        #     df = get_each_company_daily_price(self.raw_price_data, company_symbol)
        #     # print(df)
        #     # df 為存放單一公司所有日期的開高低收量資料(col小寫)
        #     result_series = eval('abstract.'+indname+'(df)')

        #     result_df[company_symbol] = result_series

        # print(result_df)

        # df = get_each_company_daily_price(self.raw_price_data, '8905')
        # result_series = eval('abstract.'+indname+'(df)')
        # print(result_series)


if __name__ == "__main__":
    data = Data()
    # 測試輸出股價資料
    # close = data.get("price:close")
    # print("收盤價:", close)
    # 測試輸出財報資料
    # roe = data.get("report:roe, EPS")
    # print(roe)
    # rsi = data.indicator('RSI')
    # print(rsi)
    # rsi.to_csv('./OutputFile/self_rsi.csv')

    # price = data.handle_price_data()
    # print(price)

    # all_companys = get_all_company_symbol(data.raw_price_data)
    # print(all_companys)

    a, b, c = data.indicator("MACD")
    a
    b
    c

    a = data.indicator("CCI")
    type(a)
    a
