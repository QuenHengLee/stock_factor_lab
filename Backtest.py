import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class Backtest():
    def __init__(self, position, init_portfolio_value = 10**6,  position_limit=1, fee_ratio=1.425/1000, tax_ratio=3/1000):
        # 初始金額
        self.init_portfolio_value = init_portfolio_value

        # position
        self.position = position
        self.position_limit = position_limit
        self.position.index = pd.to_datetime(self.position.index)

        # 取得股價資料
        self.stock = self.get_stock_data()
        # 取得有買進的訊號，只要任一股票有買進訊號，signal就會是True
        self.stock["signal"] = self.position.any(axis=1)

        # 買:手續費、賣:手續費 + 交易稅
        self.buy_extra_cost = fee_ratio
        self.sell_extra_cost = fee_ratio + tax_ratio

        # 初始化資產
        self.assets = pd.DataFrame(index=self.stock.index, columns=["portfolio_value", "cost", "init", "remain"])
        self.assets["init"] = self.init_portfolio_value
        self.shares_df = pd.DataFrame(0, index=self.stock.index, columns=self.stock.columns.drop(["signal"]))
        self.prev_values = {}

        self.calc_weighted_positions()
        self.calculate_assets()
        self.stock_data = self.create_stock_data()

    def get_stock_data(self):
        stock = pd.read_csv('./Data/test/股價.csv').set_index('date')
        # 讓日期格式一致
        stock.index = pd.to_datetime(stock.index, format='%Y/%m/%d')
        stock.ffill(inplace=True)
        stock = stock.asfreq('D', method='ffill')
        stock = stock.loc[stock.index.isin(self.position.index)]
        return stock

    def calc_weighted_positions(self):  # 計算權重
        # 統一日期
        self.total_weight = self.position.abs().sum(axis=1)
        self.position = self.position.div(self.total_weight.where(self.total_weight != 0, np.nan), axis = 0) \
                        .fillna(0).clip(-abs(self.position_limit), abs(self.position_limit))

    def calculate_assets(self):
        first_trading = True
        for day in self.stock.index:
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
                    weight_difference = self.position.loc[day] - self.position.shift(1).loc[day]
                    buy_stock_shares_list = {}
                    sell_stock_shares_list = {}

                    # 如果權重不變就重新計算portfolio value，其餘照舊
                    if abs(weight_difference).sum() == 0 : 
                        self.shares_df.loc[day] = self.shares_df.shift(1).loc[day]
                        self.assets.loc[day, "portfolio_value"] = (self.stock.loc[day].drop(['signal']) * self.shares_df.loc[day]).sum()
                        self.assets.loc[day, ["cost", "init", "remain"]] = self.assets.shift(1).loc[day, ["cost", "init", "remain"]]

                        self.prev_values = self.shares_df.loc[day].to_dict()

                        continue
                    else:
                        for s, w in weight_difference.items():
                            self.shares_df.loc[day, s] = self.prev_values[s]
                            if w < 0:
                                # 取得要賣的股票張數 e.g {'2330':19995}
                                sell_stock_shares_list[s] = np.floor((abs(w) / self.position.shift(1).loc[day][s]) * self.prev_values[s])
                            elif w > 0:
                                buy_stock_shares_list[s] = w

                        # 把要賣掉的股票*當天收盤價，扣掉手續費&交易稅，加總後就是當次可投入金額
                        sell_money = ((self.stock.loc[day].drop(['signal']) * pd.Series(sell_stock_shares_list)).sum() * (1 - self.sell_extra_cost)) + self.assets.shift(1).loc[day, "remain"]
                          
                        buy_money_per_stock = sell_money / len(buy_stock_shares_list)
                        remain=0
                        for s, w in buy_stock_shares_list.items():
                            self.shares_df.loc[day, s] = np.floor(buy_money_per_stock / (self.stock.loc[day, s] * (1 + self.buy_extra_cost)))
                            stock_price = self.stock.loc[day, s]
                            if not pd.isna(stock_price):
                                remain += self.shares_df.loc[day, s] * stock_price * (1 + self.buy_extra_cost)
                            self.shares_df.loc[day, s] += self.prev_values[s]

                        # 計算剩餘(remain)，寫在這邊是因為這時候shares_df還沒把賣掉的放進去
                        # 用投入金額 - 買的股票金額
                        remain = sell_money - remain

                        # 計算賣掉後剩幾張股票
                        for s, w in sell_stock_shares_list.items():
                            self.shares_df.loc[day, s] = self.prev_values[s] - w

                        sell_cost = ((self.shares_df.shift(1).loc[day] - self.shares_df.loc[day]) * self.stock.loc[day].drop(['signal']) * self.sell_extra_cost).sum()
                        buy_cost = (self.shares_df.loc[day] * self.stock.loc[day].drop(['signal']) * self.buy_extra_cost).sum()
                        total_cost = sell_cost + buy_cost

                        portfolio_value = (self.stock.loc[day].drop(['signal']) * self.shares_df.loc[day]).sum()

                        self.prev_values = self.shares_df.loc[day].to_dict()

                self.assets.loc[day, "portfolio_value"] = portfolio_value
                self.assets.loc[day, ["cost", "init", "remain"]] = [total_cost, sell_money, remain]

    def create_stock_data(self):
        stock_data = self.stock.copy()
        stock_data['portfolio_value'] = self.assets['portfolio_value']
        
        # 要回測的股票資料
        stocks = list(self.position.columns)

        # log return
        stock_data['portfolio_returns'] = np.log(stock_data['portfolio_value'].astype('float')).diff(1)

        for s in stocks:
            stock_data[f'{s}_shares'] = self.shares_df[s]
            stock_data[f'{s}_value'] = self.shares_df[s] * self.stock[s]
            stock_data[f'{s}_returns'] = np.divide((stock_data[s] - stock_data[s].shift(1)), stock_data[s].shift(1))
  
        stock_data.fillna(0, inplace=True)
        stock_data.replace([np.inf], 0, inplace=True)

        return stock_data

    def returns_plot(self):
        stocks = list(self.position.columns)

        # Create subplot layout
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Portfolio Returns', 'Asset Returns', 'Shares Holding per Asset', 'Weights per Asset'))

        # Add traces to the subplots
        fig.add_trace(go.Scatter(x=self.stock_data.index, y=np.exp(np.cumsum(self.stock_data['portfolio_returns']))-1, name='Portfolio'), row=1, col=1)

        for s in stocks:
            fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.stock_data[f'{s}_returns'].cumsum(), name=f'{s}'), row=1, col=2)
            fig.add_trace(go.Scatter(x=self.shares_df.index, y=self.shares_df[s], name=f'{s}'), row=2, col=1)
            fig.add_trace(go.Scatter(x=self.stock_data.index, y=self.position[s], name=f'{s}'), row=2, col=2)

        # Update subplot layout
        fig.update_layout(height=800, width=1200, title='Strategy Overview', showlegend=False)

        # Display the plot
        fig.show()