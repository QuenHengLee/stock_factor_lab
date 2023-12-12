import pandas as pd
import numpy as np
import plotly.graph_objs as go
import report
from plotly.subplots import make_subplots
from get_data import Data

def get_stock_data(position, data):
    '''
    根據position裡面的columns(股票代號)從DB中取得資料
    input:
        position
    return:
        stock : 根據position的index取得該日期的股價
        stock_price : 每一天的股價
    '''
    # 實際收盤價資料
    if data:
        all_close = data.get("price:close")
    else:
        data = Data()
        all_close = data.get("price:close")
    all_close.index = pd.to_datetime(all_close.index, format="%Y-%m-%d")
    df_dict = {}
    for symbol, p in position.items():
        start = p.index[0]
        end = p.index[-1]
        all_close = all_close[start:end]
        df_dict[symbol] = all_close[symbol]
    # print("self.df_dict:", self.df_dict)

    # 原本的DF 有開高低收量
    stock = pd.concat(
        [df_dict[symbol] for symbol in position.columns.tolist()],
        axis=1,
        keys=position.columns.tolist(),
    )
    # stock = pd.read_csv('../Data/Finlab/stock.csv').set_index('date')
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

def sim(position, resample='D', init_portfolio_value = 10**6,  position_limit=1, fee_ratio=1.425/1000, tax_ratio=3/1000, data=None):
    # 初始金額
    # self.init_portfolio_value = init_portfolio_value
    position = position_resample(position, resample)
    position = calc_weighted_positions(position, position_limit)

    # 取得股價資料
    stock_price, stock = get_stock_data(position,data)  
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
                # prev_values = shares_df.loc[day].to_dict()

                assets.loc[day, "portfolio_value"] = 0
                assets.loc[day, "cost"] = 0
                assets.loc[day, "remain"] =  0

            else:  
                first_trading = True
                if not prev_values:    #如果prev_values沒有值 -> 代表還未出現交易
                    assets.loc[day] = assets.shift(1).loc[day]
                else: #有過交易，但出現全部都是False，清空倉位
                    sell_money = ((stock.loc[day].drop(['signal']) * pd.Series(prev_values)).sum() * (1 - sell_extra_cost)) + assets.shift(1).loc[day, "remain"]
                    sell_cost = (stock.loc[day].drop(['signal']) * pd.Series(prev_values)).sum() * sell_extra_cost
                    remain = 0
                    assets.loc[day, "portfolio_value"] = sell_money
                    init_portfolio_value = sell_money
                    assets.loc[day, ["cost", "init", "remain"]] = [sell_cost, sell_money, remain]
                    prev_values={}


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
    stock_data['portfolio_value'] = assets['portfolio_value'].asfreq('D', method='ffill')
    start_trading_day = stock_data.loc[stock_data['portfolio_value'] != 0.0].index[0]
    stock_data = stock_data.loc[start_trading_day:]

    # daily return
    stock_data['portfolio_returns'] = stock_data['portfolio_value'].pct_change(1)
    stock_data = stock_data.fillna(0).replace([np.inf, -np.inf], 0)

    # 累計報酬
    stock_data['cum_returns'] = stock_data['portfolio_returns'].add(1).cumprod().sub(1)

    # 每日入選股票數量
    stock_data['company_count'] = (position != 0).sum(axis=1)
    stock_data['company_count'] = stock_data['company_count'].fillna(0)
    
    r = report.Report(stock_data, position)

    return r
