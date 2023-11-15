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
        stock_price = stock.asfreq("D", method="ffill")
        stock = stock.asfreq("D", method="ffill")
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


def get_stock_data(position):
    '''
    根據position裡面的columns(股票代號)從DB中取得資料
    input:
        position
    return:
        stock : 根據position的index取得該日期的股價
        stock_price : 每一天的股價
    '''
    # 實際收盤價資料
    data = Data()
    all_close = data.get("price:close")
    all_close.index = pd.to_datetime(all_close.index, format="%Y-%m-%d")
    df_dict = {}
    for symbol, position in position.items():
        start = position.index[0]
        end = position.index[-1]
        all_close = all_close[start:end]
        df_dict[symbol] = all_close[symbol]
    # print("self.df_dict:", self.df_dict)

    # 原本的DF 有開高低收量
    stock = pd.concat(
        [df_dict[symbol] for symbol in position.columns.tolist()],
        axis=1,
        keys=position.columns.tolist(),
    )
    # 讓日期格式一致
    stock.index = pd.to_datetime(stock.index, format="%Y-%m-%d")
    stock.ffill(inplace=True)
    stock_price = stock.asfreq("D", method="ffill")
    stock = stock.asfreq("D", method="ffill")
    stock = stock.loc[stock.index.isin(position.index)]
    return stock_price, stock

def calc_weighted_positions(position, position_limit):
    '''
    根據「等權重」資金配置法，將position轉變為每支股票要投入的%數

    Args:
        position_limit:可以限制單一股票最高要投入幾%資金，控制風險用

    return:
        weighted_position: 加權過後的position

    Exapmle:
        原始position:
            |            | Stock 2330 | Stock 1101 | Stock 2454 | Stock 2540 |
            |------------|------------|------------|------------|------------|
            | 2021-12-31 | True       | False      | False      | True       |
            | 2022-03-31 | True       | True       | True       | False      |
            | 2022-06-30 | False      | True       | False      | False      |
        
        加權過後position:
            |            | Stock 2330 | Stock 1101 | Stock 2454 | Stock 2540 |
            |------------|------------|------------|------------|------------|
            | 2021-12-31 | 0.5        | 0          | 0          | 0.5        |
            | 2022-03-31 | 0.25       | 0.25       | 0.25       | 0          |
            | 2022-06-30 | 0          | 1          | 0          | 0          |
    '''
    position.index = pd.to_datetime(position.index)

    # 統一日期
    total_weight = position.abs().sum(axis=1)
    position = position.div(total_weight.where(total_weight != 0, np.nan), axis = 0) \
                    .fillna(0).clip(-abs(position_limit), abs(position_limit))
    
    return position

def position_resample(position, resample):
    '''
    根據想要換股/再平衡的週期調整position
    '''
    position.index = pd.to_datetime(position.index, format='%Y-%m-%d')
    # 先將position中按照想要輪動股票的週期排序
    position = position.asfreq(resample, method='ffill')

    return position

def sim(self, position, resample='D', init_portfolio_value = 10**6,  position_limit=1, fee_ratio=1.425/1000, tax_ratio=3/1000):
    # 初始金額
    # self.init_portfolio_value = init_portfolio_value
    position = position_resample(position, resample)
    position = calc_weighted_positions(position, position_limit)

    # 取得股價資料
    stock_price, stock = get_stock_data(position)  
    # # 取得有買進的訊號，只要任一股票有買進訊號，signal就會是True
    stock["signal"] = position.any(axis=1)

    # 買:手續費、賣:手續費 + 交易稅
    buy_extra_cost = fee_ratio
    sell_extra_cost = fee_ratio + tax_ratio

    # 初始化資產
    assets = pd.DataFrame(index=position.index, columns=["portfolio_value", "cost", "init", "remain"])
    assets["init"] = init_portfolio_value
    shares_df = pd.DataFrame(0, index=position.index, columns=position.columns)
    prev_values = {}

    first_trading = True
    for day in position.index:
        # 持有股票
        if stock.loc[day]['signal'] == False:
            if day == stock.index[0]:
                # 第一天直接取shares_df裡的值 (全部都0)
                prev_values = shares_df.loc[day].to_dict()

                assets.loc[day, "portfolio_value"] = 0
                assets.loc[day, "cost"] = 0
                assets.loc[day, "remain"] =  0

            else:   # 剩下的取前一天的
                shares_df.loc[day] = shares_df.shift(1).loc[day]
                # 只有portfolio value重新計算
                assets.loc[day, "portfolio_value"] = ((stock.loc[day].drop(["signal"]) * shares_df.loc[day]).sum())
                assets.loc[day, ["cost", "init", "remain"]] = assets.shift(1).loc[day, ["cost", "init", "remain"]]
                position.loc[day] = position.shift(1).loc[day]


        # 再平衡（rebalance）
        else : 
            if first_trading:
                first_trading = False
                buy_amount = init_portfolio_value * position.loc[day] / (stock.loc[day].drop(['signal']) * (1 + buy_extra_cost))
                shares_df.loc[day] = np.floor(buy_amount.fillna(0.0))

                portfolio_value = (stock.loc[day].drop(['signal']) * shares_df.loc[day]).sum()
                total_cost = (stock.loc[day].drop(['signal']) * shares_df.loc[day] * buy_extra_cost).sum()
                remain = init_portfolio_value - portfolio_value - total_cost
                sell_money = init_portfolio_value

                prev_values = shares_df.loc[day].to_dict()
            else:
                # 把前一次持有的股票全部賣掉，換新的股票進場
                # 可以拿來投資的金額 = 股價 * 前一次持有之張數 * 賣出手續費 + 前一次剩餘的金額 
                sell_money = ((stock.loc[day].drop(['signal']) * pd.Series(prev_values)).sum() * (1 - sell_extra_cost)) + assets.shift(1).loc[day, "remain"]

                # 計算每一支股票買多少張
                shares_df.loc[day] = np.floor(sell_money * position.loc[day] / (stock.drop(['signal'], axis=1).loc[day] * (1 + buy_extra_cost)))

                sell_cost = (stock.loc[day].drop(['signal']) * pd.Series(prev_values)).sum() * sell_extra_cost
                buy_cost = (shares_df.loc[day] * stock.loc[day].drop(['signal']) * buy_extra_cost).sum()
                total_cost = sell_cost + buy_cost

                # 用投入金額 - 購入股票金額，可以得到因無條件捨去後剩餘的金額
                remain = sell_money - (shares_df.loc[day] * stock.drop(['signal'], axis=1).loc[day] * (1+buy_extra_cost)).sum()

                portfolio_value = (stock.loc[day].drop(['signal']) * shares_df.loc[day]).sum()
                prev_values = shares_df.loc[day].to_dict()

            assets.loc[day, "portfolio_value"] = portfolio_value
            assets.loc[day, ["cost", "init", "remain"]] = [total_cost, sell_money, remain]

    stock_data = pd.DataFrame(index=stock_price.index)
    shares_df = shares_df.reindex(stock_price.index, method="ffill")
    stock_data['portfolio_value'] = (stock_price * shares_df).sum(axis=1)

    # daily return
    stock_data['portfolio_returns'] = stock_data['portfolio_value'].pct_change(1)
    stock_data = stock_data.fillna(0).replace([np.inf, -np.inf], 0)

    # 累計報酬
    stock_data['cum_returns'] = stock_data['portfolio_returns'].add(1).cumprod()

    return stock_data

# 用來安全進行除法的函數。如果分母 d 不等於零，則返回 n / d，否則返回 0。
def safe_division(n, d):
    return n / d if d else 0

# 用來將dataframe按照月份排列
def sort_month(df):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return df[month_order]