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

def indicator(indname, adjust_price=False, resample='D', market:Union[str, MarketInfo]='TW_STOCK', **kwargs):
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
    package = None

    try:
        from talib import abstract
        import talib
        attr = getattr(abstract, indname)
        package = 'talib'
    except:
        try:
            import pandas_ta
            # test df.ta has attribute
            getattr(pd.DataFrame().ta, indname)
            attr = lambda df, **kwargs: getattr(df.ta, indname)(**kwargs)
            package = 'pandas_ta'
        except:
            raise Exception(
                "Please install TA-Lib or pandas_ta to get indicators.")


    market = get_market_info(user_market_info=market)

    close = market.get_price('close', adj=adjust_price)
    open_ = market.get_price('open', adj=adjust_price)
    high = market.get_price('high', adj=adjust_price)
    low = market.get_price('low', adj=adjust_price)
    volume = market.get_price('volume', adj=adjust_price)

    if resample.upper() != 'D':
        close = close.resample(resample).last()
        open_ = open_.resample(resample).first()
        high = high.resample(resample).max()
        low = low.resample(resample).min()
        volume = volume.resample(resample).sum()

    dfs = {}
    default_output_columns = None
    for key in close.columns:

        prices = {'open': open_[key].ffill(),
                  'high': high[key].ffill(),
                  'low': low[key].ffill(),
                  'close': close[key].ffill(),
                  'volume': volume[key].ffill()}

        if package == 'pandas_ta':
            prices = pd.DataFrame(prices)
            s = attr(prices, **kwargs)

        elif package == 'talib':
            abstract_input = list(attr.input_names.values())[0]
            abstract_input = get_input_args(attr)

            # quick fix talib bug
            if indname == 'OBV':
                abstract_input = ['close', 'volume']

            if indname == 'BETA':
                abstract_input = ['high', 'low']

            if isinstance(abstract_input, str):
                abstract_input = [abstract_input]
            paras = [prices[k] for k in abstract_input]
            s = attr(*paras, **kwargs)
        else:
            raise Exception("Cannot determine technical package from indname")

        if isinstance(s, list):
            s = {i: series for i, series in enumerate(s)}

        if isinstance(s, np.ndarray):
            s = {0: s}

        if isinstance(s, pd.Series):
            s = {0: s.values}

        if isinstance(s, pd.DataFrame):
            s = {i: series.values for i, series in s.items()}

        if default_output_columns is None:
            default_output_columns = list(s.keys())

        for colname, series in s.items():
            if colname not in dfs:
                dfs[colname] = {}
            dfs[colname][key] = series if isinstance(
                series, pd.Series) else series

    newdic = {}
    for key, df in dfs.items():
        newdic[key] = pd.DataFrame(df, index=close.index)

    ret = [newdic[n] for n in default_output_columns]
    ret = [d.apply(lambda s:pd.to_numeric(s, errors='coerce')) for d in ret]

    if len(ret) == 1:
        return finlab.dataframe.FinlabDataFrame(ret[0])

    return tuple([finlab.dataframe.FinlabDataFrame(df) for df in ret])


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
