from database import Database
import talib
import pandas as pd
import numpy as np
from talib import MA_Type


# Abstract API：
from talib import abstract
class Data():

    # 接收SQL下來DB的資料
    def __init__(self):
        # 與資料庫連線
        # 下載全部資料(from database)
        self.db = Database()
        self.sql_data = self.db.get_daily_stock()
        self.format_daily_stock()

    '''
    INPUT: self, dataset(str)帶入想要的資料名稱Ex. price:close、report:roe
    OUTPUT: 一個內容為所有公司股價/財報相關資料的Dataframe
    FUNCTION: 從DB中取得資料
    '''

    # 將所有資料分類
    def get(self,dataset):
        selected_data = self.sql_data[['date',dataset,'company_symbol']]
        pivot_data = selected_data.pivot_table(index='date', columns='company_symbol', values=dataset)
        return pivot_data
      
    def format_daily_stock(self):
        self.all_open = self.get('open')
        self.all_high = self.get('high')
        self.all_low = self.get('low')
        self.all_close = self.get('close')
        self.all_volume = self.get('volume')
        self.all_market_capital = self.get('market_capital')
        # 宣告一個字典存放這些dataframe
        # 呼叫.indicator時，就是傳入這個dict
        self.all_data_dict = {'open': self.all_open, 
                              'high': self.all_high, 
                              'low': self.all_low, 
                              'close': self.all_close, 
                              'volume': self.all_volume,
                              'market_capital': self.all_market_capital}
        return  self.all_data_dict
            
    def get_daily_stock_list(self,column):
         
        open_prices = self.all_data_dict['open'][column].values
        high_prices = self.all_data_dict['high'][column].values
        low_prices = self.all_data_dict['low'][column].values
        close_prices = self.all_data_dict['close'][column].values
        volume = self.all_data_dict['volume'][column].values
        market_capital = self.all_data_dict['market_capital'][column].values
        inputs = {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'market_capital': market_capital
        }
        inputs = pd.DataFrame(inputs)
        return inputs


 
                


if __name__ == "__main__":
    Data = Data()
    # SelectIndex.get('open')
    output = Data.get_indicator_value('BBANDS',timeperiod=25)
    print(output)
    

