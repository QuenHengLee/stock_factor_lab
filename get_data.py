from database import Database
from format_data import format_price_data, format_report_data, handle_price_data, get_each_company_daily_price,get_all_company_symbol
# import talib
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

    def indicator(self,indname, adjust_price=False, resample='D', market='TW_STOCK', **kwargs):
        """支援 Talib 和 pandas_ta 上百種技術指標，計算 2000 檔股票、10年的所有資訊。

        在使用這個函式前，需要安裝計算技術指標的 Packages

        * [Ta-Lib](https://github.com/mrjbq7/ta-lib)
        * [Pandas-ta](https://github.com/twopirllc/pandas-ta)

        Args:
            indname (str): 指標名稱，
                以 TA-Lib 舉例，例如 SMA, STOCH, RSI 等，可以參考 [talib 文件](https://mrjbq7.github.io/ta-lib/doc_index.html)。

                以 Pandas-ta 舉例，例如 supertrend, ssf 等，可以參考 [Pandas-ta 文件](https://twopirllc.github.io/pandas-ta/#indicators-by-category)。
            adjust_price (bool): 是否使用還原股價計算。
            resample (str): 技術指標價格週期，ex: `D` 代表日線, `W` 代表週線, `M` 代表月線。
            market (str): 市場選擇，ex: `TW_STOCK` 代表台股, `US_STOCK` 代表美股。
            **kwargs (dict): 技術指標的參數設定，TA-Lib 中的 RSI 為例，調整項為計算週期 `timeperiod=14`。
        建議使用者可以先參考以下範例，並且搭配 talib官方文件，就可以掌握製作技術指標的方法了。
        """
        # 先取得所有公司，因為計算指標是一間一間算
        all_company_symbol = get_all_company_symbol(data.raw_price_data)
        for company_symbol in all_company_symbol:
            df = get_each_company_daily_price(self.raw_price_data, company_symbol)
            # df 為存放單一公司所有日期的開高低收量資料(col小寫)
            result = eval('abstract.'+indname+'(df)')
            
        if isinstance(result, pd.core.frame.DataFrame):
            # 如果是DataFrame，表示有多個回傳值
            # 這裡可以動態處理不確定數量的回傳值和欄位名稱
            return [result[col_name] for col_name in result.columns]
        elif isinstance(result, pd.Series):
            # 如果是Series，表示只有一個回傳值
            # 直接將該回傳值作為單一元素回傳
            return result
            



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

    all_companys = get_all_company_symbol(data.raw_price_data)
    print(all_companys)