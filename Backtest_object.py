import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_data import Data

class Backtest:
    def __init__(self, position, position_limit=1, fee_ratio=1.425/1000, tax_ratio=3/1000):
        self.position = position
        self.position_limit = position_limit
        self.fee_ratio = fee_ratio
        self.tax_ratio = tax_ratio
        self.position.index = pd.to_datetime(self.position.index)

        # -----------------------------------------------------------------------------------------
        # 為了取得每隻股票的收盤價(Close)
        # 最終的df : index是日期，每一欄會是股票的收盤價(欄位名稱是股票代號)
        # self.df_dict = {}
        # for symbol, position in self.position.items():
        #     self.df_dict[symbol] = yf.download(symbol+".TW", start=position.index[0], end=position.index[-1])

        # self.df = pd.concat([self.df_dict[symbol] for symbol in self.position.columns.tolist()], axis=1, keys=self.position.columns.tolist())
        # self.df = self.df.xs('Close', level=1, axis=1)
        # # 讓日期格式一致
        # self.df.index = pd.to_datetime(self.df.index, format='&Y-%m-%d')


        # edit by quen
        # -----------------------------------------------------------------------------------------
        # 建立取的股票價格的物件
        gti = Data()
        all_close = gti.get('close')
        all_close.index = pd.to_datetime(all_close.index, format='%Y-%m-%d')
        # 為了取得每隻股票的收盤價(Close)
        # 最終的df : index是日期，每一欄會是股票的收盤價(欄位名稱是股票代號)
        self.df_dict = {}
        for symbol, position in self.position.items():
            start=position.index[0]
            end=position.index[-1]
            all_close = all_close[start:end]
            # for test output
            # print("symbol: ",symbol)
            # print("position: ", position)
            # close from DB
            self.df_dict[symbol] = all_close[symbol]
            
            # close from yf
            # self.df_dict[symbol] = yf.download(symbol+".TW", start=position.index[0], end=position.index[-1])
             
        # 原本的DF 有開高低收量
        self.df = pd.concat([self.df_dict[symbol] for symbol in self.position.columns.tolist()], axis=1, keys=self.position.columns.tolist())
 
        # -----------------------------------------------------------------------------------------
 
               
        # 讓日期格式一致
        self.df.index = pd.to_datetime(self.df.index, format='&Y-%m-%d')
        
        # 取得有買進的訊號，只要任一股票有買進訊號，signal就會是True
        self.df["signal"] = self.position.any(axis=1)
        print("self.df:" ,self.df) 
        self.calc_weighted_positions()
        self.get_trade_dates()
        self.profit = self.calc_profit() # ERROR MSG
        self.max_dd = self.profit.min()
        self.cumul_profit = (self.profit + 1).prod() -1
        

    def calc_weighted_positions(self):
        self.total_weight = self.position.abs().sum(axis=1)
        print('total_weight', self.total_weight)
        self.position = self.position.div(self.total_weight.where(self.total_weight != 0, np.nan), axis = 0) \
                        .fillna(0).clip(-abs(self.position_limit), abs(self.position_limit))

    def get_trade_dates(self):
        position = False
        buydates, selldates = [],[]

        for index, row in self.df.iterrows():
            print(buydates)
            print(selldates)
           
            if not position and row['signal'] == True:
                position = True
                buydates.append(index)

            if position and row["signal"] == False:
                position = False
                selldates.append(index)

        # 取得收盤價
        self.buy_df = self.df.loc[buydates].drop(["signal"], axis=1)
        self.sell_df = self.df.loc[selldates].drop(["signal"], axis=1)
        print("buydf + selldf")
        print(self.buy_df)
        print( self.sell_df)

    def calc_profit(self):
        if self.buy_df.index[-1] > self.sell_df.index[-1]:
            self.buy_df = self.buy_df[:-1]

        # 取得有交易的倉位權重分布
        self.trade_position = self.position.loc[self.buy_df.index.tolist()]
        
        # 計算手續費(買進、賣出都會有一筆，所以加起來算)
        # 把手續費 * trade_position(權重分布)然後加總(sum)，就是每一筆交易的手續費
        total_fee = (self.sell_df.values + self.buy_df.values) * self.fee_ratio
        self.fee = (total_fee * self.trade_position.values).sum(axis=1)

        # 計算交要稅(賣出)
        # 把交易稅 * trade_position(權重分布)然後加總(sum)，就是每一筆交易的交易稅
        total_tax = self.sell_df.values * self.tax_ratio
        self.tax = (total_tax * self.trade_position.values).sum(axis=1)

        # 毛利
        profit = self.sell_df.values
        # 成本
        cost = self.buy_df.values + total_fee + total_tax

        return_ratio = (((profit - cost)/self.buy_df.values) * self.trade_position.values).sum(axis=1)
        return return_ratio

