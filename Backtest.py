import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from get_data import Data

class Backtest():
    def __init__(self, position, resample='D', init_portfolio_value = 10**6,  position_limit=1, fee_ratio=1.425/1000, tax_ratio=3/1000):
        # 初始金額
        self.init_portfolio_value = init_portfolio_value
        self.position = self.position_resample(position, resample)
        self.position = self.calc_weighted_positions(self.position, position_limit)
        # self.position = self.calc_weighted_positions(position, position_limit)

        # 取得股價資料
        self.stock_price, self.stock = self.get_stock_data()  
        # # 取得有買進的訊號，只要任一股票有買進訊號，signal就會是True
        self.stock["signal"] = self.position.any(axis=1)

        # 買:手續費、賣:手續費 + 交易稅
        self.buy_extra_cost = fee_ratio
        self.sell_extra_cost = fee_ratio + tax_ratio

        # 初始化資產
        self.assets = pd.DataFrame(index=self.position.index, columns=["portfolio_value", "cost", "init", "remain"])
        self.assets["init"] = self.init_portfolio_value
        self.shares_df = pd.DataFrame(0, index=self.position.index, columns=self.position.columns)
        self.prev_values = {}
        # 執行回測
        self.sim()
        # stock_data : 投資組合價值、日回報、累計回報
        self.stock_data = self.create_stock_data()


    def get_stock_data(self):
        # stock = pd.read_csv('../Data/test/股價.csv').set_index('date')

        # 實際收盤價資料
        data = Data()
        all_close = data.get("price:close")
        all_close.index = pd.to_datetime(all_close.index, format="%Y-%m-%d")
        self.df_dict = {}
        for symbol, position in self.position.items():
            start = position.index[0]
            end = position.index[-1]
            all_close = all_close[start:end]
            self.df_dict[symbol] = all_close[symbol]
        # print("self.df_dict:", self.df_dict)

        # 原本的DF 有開高低收量
        stock = pd.concat(
            [self.df_dict[symbol] for symbol in self.position.columns.tolist()],
            axis=1,
            keys=self.position.columns.tolist(),
        )
        # 讓日期格式一致
        stock.index = pd.to_datetime(stock.index, format="%Y-%m-%d")
        stock.ffill(inplace=True)
        stock = stock.asfreq("D", method="ffill")
        stock_price = stock.asfreq("D", method="ffill")
        stock = stock.loc[stock.index.isin(self.position.index)]

        return stock_price, stock

    def calc_weighted_positions(self,position, position_limit):  # 計算權重
        position.index = pd.to_datetime(position.index)

        # 統一日期
        total_weight = position.abs().sum(axis=1)
        position = position.div(total_weight.where(total_weight != 0, np.nan), axis = 0) \
                        .fillna(0).clip(-abs(position_limit), abs(position_limit))
        
        return position

    def position_resample(self, position, resample):
        position.index = pd.to_datetime(position.index, format='%Y-%m-%d')
        # 先將position中按照想要輪動股票的週期排序
        position = position.asfreq(resample, method='ffill')

        return position

    def sim(self):
        first_trading = True
        for day in self.position.index:
            # 持有股票
            if self.stock.loc[day]['signal'] == False:
                if day == self.stock.index[0]:
                    # 第一天直接取shares_df裡的值 (全部都0)
                    self.prev_values = self.shares_df.loc[day].to_dict()

                    self.assets.loc[day, "portfolio_value"] = 0
                    self.assets.loc[day, "cost"] = 0
                    self.assets.loc[day, "remain"] =  0

                else:   # 剩下的取前一天的
                    self.shares_df.loc[day] = self.shares_df.shift(1).loc[day]
                    # 只有portfolio value重新計算
                    self.assets.loc[day, "portfolio_value"] = ((self.stock.loc[day].drop(["signal"]) * self.shares_df.loc[day]).sum())
                    self.assets.loc[day, ["cost", "init", "remain"]] = self.assets.shift(1).loc[day, ["cost", "init", "remain"]]
                    self.position.loc[day] = self.position.shift(1).loc[day]


            # 再平衡（rebalance）
            else : 
                if first_trading:
                    first_trading = False
                    buy_amount = self.init_portfolio_value * self.position.loc[day] / (self.stock.loc[day].drop(['signal']) * (1 + self.buy_extra_cost))
                    self.shares_df.loc[day] = np.floor(buy_amount.fillna(0.0))

                    portfolio_value = (self.stock.loc[day].drop(['signal']) * self.shares_df.loc[day]).sum()
                    total_cost = (self.stock.loc[day].drop(['signal']) * self.shares_df.loc[day] * self.buy_extra_cost).sum()
                    remain = self.init_portfolio_value - portfolio_value - total_cost
                    sell_money = self.init_portfolio_value

                    self.prev_values = self.shares_df.loc[day].to_dict()
                else:
                    # 把前一次持有的股票全部賣掉，換新的股票進場
                    # 可以拿來投資的金額 = 股價 * 前一次持有之張數 * 賣出手續費 + 前一次剩餘的金額 
                    sell_money = ((self.stock.loc[day].drop(['signal']) * pd.Series(self.prev_values)).sum() * (1 - self.sell_extra_cost)) + self.assets.shift(1).loc[day, "remain"]

                    # 計算每一支股票買多少張
                    self.shares_df.loc[day] = np.floor(sell_money * self.position.loc[day] / (self.stock.drop(['signal'], axis=1).loc[day] * (1 + self.buy_extra_cost)))

                    sell_cost = (self.stock.loc[day].drop(['signal']) * pd.Series(self.prev_values)).sum() * self.sell_extra_cost
                    buy_cost = (self.shares_df.loc[day] * self.stock.loc[day].drop(['signal']) * self.buy_extra_cost).sum()
                    total_cost = sell_cost + buy_cost

                    # 用投入金額 - 購入股票金額，可以得到因無條件捨去後剩餘的金額
                    remain = sell_money - (self.shares_df.loc[day] * self.stock.drop(['signal'], axis=1).loc[day] * (1+self.buy_extra_cost)).sum()

                    portfolio_value = (self.stock.loc[day].drop(['signal']) * self.shares_df.loc[day]).sum()
                    self.prev_values = self.shares_df.loc[day].to_dict()

                self.assets.loc[day, "portfolio_value"] = portfolio_value
                self.assets.loc[day, ["cost", "init", "remain"]] = [total_cost, sell_money, remain]

    def create_stock_data(self):
        stock_data = pd.DataFrame(index=self.stock_price.index)
        self.shares_df = self.shares_df.reindex(self.stock_price.index, method="ffill")
        stock_data['portfolio_value'] = (self.stock_price * self.shares_df).sum(axis=1)
        
        # 要回測的股票資料
        stocks = list(self.position.columns)

        # daily return
        stock_data['portfolio_returns'] = stock_data['portfolio_value'].pct_change(1)
        stock_data.fillna(0, inplace=True)
        stock_data.replace([np.inf, -np.inf], 0, inplace=True)
        
        # 累計報酬
        stock_data['cum_returns'] = stock_data['portfolio_returns'].add(1).cumprod()

        return stock_data
    
    def calc_mdd(self):
        '''
        計算Drawdown的方式是找出截至當下的最大累計報酬(%)除以當下的累計報酬
        所以用累計報酬/累計報酬.cummax()

        return:
            dd : 每天的drawdown (可用來畫圖)
            mdd : 最大回落
            start : mdd開始日期
            end : mdd結束日期
            days : 持續時間
        '''
        r = self.stock_data['cum_returns']
        dd = r.div(r.cummax()).sub(1)
        mdd = dd.min()
        end = dd.idxmin()
        start = r.loc[:end].idxmax()
        days = end-start

        return dd, mdd, start, end, days
    
    def calc_cagr(self):
        '''
        計算CAGR用(最終價值/初始價值)^(1/持有時間(年))-1
        那其實最終價值/初始價值就跟累計回報會差不多，
        所以公式可以變為:累計報酬^(1/持有時間(年))-1

        return:
            cagr : 年均報酬率
        '''
        first_day = self.stock_data[self.stock_data['portfolio_value']!=0].idxmin()[0]
        # 計算持有幾年
        num_years = safe_division(365.25, (self.stock_data.index[-1] - first_day).days)

        return np.power(self.stock_data.iloc[-1]['cum_returns'], num_years)-1

    def calc_monthly_return(self):
        '''
        monthly_return : 月報酬
        公式 : 本月投資組合價值/上個月的投資組合價值
        可以用pct_change()
        最後利用樞紐整理成dataframe
        
        return:
            dataframe : index:年分、columns:月份
        '''
        # 先copy一份原始資料
        stock_data = pd.DataFrame(self.stock_data['portfolio_value'].resample('M').last())
        stock_data['monthly_returns'] = stock_data['portfolio_value'].pct_change()

        #  columns : 'year' 和 'month' 
        stock_data['year'] = stock_data.index.year
        stock_data['month'] = stock_data.index.strftime('%b')  # 將月份轉換為縮寫形式

        return round(sort_month(stock_data.pivot_table(index='year', columns='month', values='monthly_returns').replace([np.inf, -np.inf, np.nan], 0))*100, 1)
    
    def calc_yearly_return(self):
        '''
        yearly_return : 年回報
        公式 : 今年投資組合價值/去年的投資組合價值
        可以用pct_change()，但因第一年會有沒有值的情況，
        因此先計算【日回報】，再利用(1+r1)*(1+r2)*...*(1+rn)-1來計算每年回報

        return:
            dataframe: columns:年分
        '''
        # 先前已經計算過日回報
        daily_returns = pd.DataFrame(self.stock_data['portfolio_returns'])

        # 使用 resample 計算每年的年報酬
        annual_returns = daily_returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
        annual_returns = annual_returns[annual_returns['portfolio_returns']!=0]

        # 將index改為年分
        annual_returns.index = annual_returns.index.year

        return round(annual_returns.T * 100, 1)



# 用來安全進行除法的函數。如果分母 d 不等於零，則返回 n / d，否則返回 0。
def safe_division(n, d):
    return n / d if d else 0

# 用來將dataframe按照月份排列
def sort_month(df):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return df[month_order]