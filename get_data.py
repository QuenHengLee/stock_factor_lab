from database import Database
from finlab_data_frame import CustomDataFrame
from format_data import *
import talib
import pandas as pd
import numpy as np

# Abstract API：
from talib import abstract


class Data:
    def __init__(self):
        """
        初始化物件，連接資料庫並下載相關資料。

        Args:
            self: 類的實例（通常是類的物件，不需要額外指定）。
        """
        # 連接資料庫並下載股價資料
        self.db = Database()
        self.raw_price_data = self.db.get_daily_stock()
        self.all_price_dict = self.handle_price_data()
        # 下載財報資料
        self.raw_report_data = self.db.get_finance_report()

    def format_price_data(self, item):
        """
        從原始股價資料中處理特定項目的資料。

        Args:
            item (str): 要處理的項目名稱。

        Returns:
            pandas.DataFrame: 處理後的股價資料的DataFrame。
        """
        return format_price_data(self.raw_price_data, item)

    def format_report_data(self, factor):
        """
        從原始財報資料中處理特定因子的資料。

        Args:
            factor (str): 要處理的財報因子名稱。

        Returns:
            pandas.DataFrame: 處理後的財報資料的DataFrame。
        """
        return format_report_data(self.raw_report_data, factor)

    def handle_price_data(self):
        """
        處理原始股價資料並返回一個字典。

        Returns:
            dict: 包含處理後的股價資料的字典。
        """
        return handle_price_data(self.raw_price_data)

    # 取得資料起點
    def get(self, dataset):
        """
        從資料庫中取得資料並返回相應的DataFrame。

        Args:
            self: 類的實例（通常是類的物件，不需要額外指定）。
            dataset (str): 想要的資料名稱，格式如 "price:close" 或 "report:roe"。

        Returns:
            pandas.DataFrame or dict: 一個包含所有公司股價/財報相關資料的DataFrame 或一個包含多個財報資料的字典。

        註解:
        - 此方法根據輸入的資料名稱（dataset）擷取相對應的資料。
        - 如果資料名稱是 "price"，則返回股價相關的DataFrame。
        - 如果資料名稱是 "report"，則返回多個財報資料的字典，每個財報資料對應一個鍵。
        - 資料名稱應以冒號分隔，例如 "price:close" 或 "report:roe"。
        - 如果輸入格式不正確，將列印錯誤訊息。

        使用示例:
        ```
        # 創建類的實例
        my_instance = YourClass()
        # 呼叫get方法取得股價資料
        price_data = my_instance.get("price:close")
        # price_data 可能是一個DataFrame，包含股價相關資料。
        # 或者呼叫get方法取得多個財報資料
        report_data = my_instance.get("report:roe,eps")
        # report_data 是一個字典，包含多個財報資料，例如 {"roe": DataFrame, "eps": DataFrame}。
        ```
        """
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
        """
        計算指標並回傳相關數據。

        Args:
            self: 類的實例（通常是類的物件，不需要額外指定）。
            indname (str): 要計算的指標名稱。
            adjust_price (bool): 是否進行價格調整（預設為False）。
            resample (str): 重新取樣的頻率（預設為"D"，即每日）。
            market (str): 市場類別（預設為"TW_STOCK"，即台灣股市）。
            **kwargs: 額外的參數，用於傳遞給其他函數。

        Returns:
            pandas.DataFrame or tuple of DataFrames: 根據指標回傳值的數量，返回一個DataFrame或包含多個DataFrames的元組。每個DataFrame 包含指標計算的結果。

        功能:
        1. 先取得所有公司的代號列表（all_company_symbol）。
        2. 根據第一家公司的代號，計算指標將返回的數量（num_of_return）。
        3. 創建一個元組（dataframe_tuple）以保存計算結果的DataFrames。
        4. 使用巢狀迴圈計算每家公司的指標值並填充到相應的DataFrames 中。
        5. 根據回傳值的數量，返回單個DataFrame 或包含多個DataFrames 的元組。

        註解:
        - 這個方法的主要功能是計算給定指標（indname）的數值，對每家公司進行計算。
        - 指標的計算結果可以包含多個值，這些值保存在不同的DataFrames 中。
        - 回傳值的數量（num_of_return）決定了回傳的數據結構。如果只有一個值，將返回單個DataFrame。
        - 如果有多個值，將返回一個包含這些DataFrames 的元組。

        使用示例:
        ```
        # 創建類的實例
        my_instance = YourClass()
        # 呼叫indicator方法計算指標
        result = my_instance.indicator("SMA", adjust_price=True)
        # result 可能是一個DataFrame 或多個DataFrames 的元組，視指標計算的結果而定。
        ```
        """

        # 先處理一下計算indicator需要用到的kwargs
        # 使用列表推導式將字典轉換為字串形式的鍵值對
        key_value_pairs = [f"{key}={value}" for key, value in kwargs.items()]
        # 使用逗號和空格將鍵值對連接起來
        kwargs_result_str = ", ".join(key_value_pairs)
        kwargs_result_str = "," + kwargs_result_str
        # 打印結果
        # print(kwargs_result_str)

        # 先取得所有公司list，因為計算指標是一間一間算
        all_company_symbol = get_all_company_symbol(self.raw_price_data)

        # 先計算該指標會回傳幾個值
        tmp_company_daily_price = get_each_company_daily_price(
            self.raw_price_data, all_company_symbol[0]
        )
        num_of_return = get_number_of_indicator_return(indname, tmp_company_daily_price)

        # 再根據回傳值的數量動態宣告N個dataframe在一個tuple中
        empty_dataframe = CustomDataFrame()
        dataframe_tuple = tuple()
        # # 複製空的df到tuple中
        # for _ in range(num_of_return):
        #     df = empty_dataframe.copy()
        #     dataframe_tuple += (df,)

        # 创建一个空的主字典
        nested_dict = {}
        # 用巢狀迴圈逐一填入N公司的M個回傳指標資料
        for i in range(num_of_return):
            nested_dict[i] = {}  # 创建一个嵌套字典
            for company_symbol in all_company_symbol:
                df = get_each_company_daily_price(self.raw_price_data, company_symbol)
                # df 為存放單一公司所有日期的開高低收量資料(col小寫)
                tmp_eval_str = "abstract." + indname + "(df" + kwargs_result_str + ")"
                # print("執行字串: ", tmp_eval_str)
                result = eval(tmp_eval_str)
                # 假如只有回傳一個值，回以series呈現，這邊要轉成dataframe
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                else:
                    pass  # df不用再轉df
                # 這種方法組合DF，會導致過度碎片化
                print(result)
                # dataframe_tuple[i][company_symbol] = result.iloc[:, i]
                # subkey = company_symbol
                # value = result.iloc[:, i]
                # nested_dict[i][subkey] = value

        # # 複製空的df到tuple中
        # for _ in range(num_of_return):
        #     # 合并所有的1-D DataFrame成一个2D DataFrame
        # merged_dataframe = pd.concat(dataframes_dict.values(), axis=1)

        if num_of_return == 1:
            return dataframe_tuple[0]
        else:
            return nested_dict


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

    a = adx = data.indicator("ADX")
    a

    b = adx = data.indicator("ADX", timeperiod=50)
    b
