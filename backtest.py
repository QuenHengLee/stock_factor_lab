import sys
import warnings
import datetime
import numpy as np
import pandas as pd
from typing import Union
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import report
from get_data import Data
from core.backtest_core import backtest_, get_trade_stocks

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
    # stock = pd.read_csv('../Data/Finlab/stock.csv').set_index('date')
    # 讓日期格式一致
    stock.index = pd.to_datetime(stock.index, format="%Y-%m-%d")
    stock_price = stock.asfreq("D", method="ffill")
    stock = stock_price.reindex(position.index, method="ffill")
    stock_price['cash'] = 1
    # stock = stock.asfreq("D", method="ffill")
    # stock = stock.loc[stock.index.isin(position.index)]

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
    # position = (position * stock) >0

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
    # position = position.asfreq(resample, method='ffill')
    position = position.resample(resample).last().fillna(method='ffill')

    return position

def warning_resample(resample):

  if '+' not in resample and '-' not in resample:
      return

  if '-' in resample and not resample.split('-')[-1].isdigit():
      return

  if '+' in resample:
      r, o = resample.split('+')
  elif '-' in resample:
      r, o = resample.split('-')

  warnings.warn(f"The argument sim(..., resample = '{resample}') will no longer support after 0.1.37.dev1.\n"
                f"please use sim(..., resample='{r}', offset='{o}d')", DeprecationWarning)

def calc_essential_price(price, dates):

    dt = min(price.index.values[1:] - price.index.values[:-1])

    indexer = price.index.get_indexer(dates + dt)

    valid_idx = np.where(indexer == -1, np.searchsorted(price.index, dates, side='right'), indexer)
    valid_idx = np.where(valid_idx >= len(price), len(price) - 1, valid_idx)

    return price.iloc[valid_idx]

def arguments(price, high, low, open_, position, resample_dates=None, fast_mode=False):

    resample_dates = price.index if resample_dates is None else resample_dates
    position = position.astype(float).fillna(0)

    if fast_mode:
        date_index = pd.to_datetime(resample_dates)
        position = position.reindex(date_index, method='ffill')
        price = calc_essential_price(price, date_index)
        high = calc_essential_price(high, date_index)
        low = calc_essential_price(low, date_index)
        open_ = calc_essential_price(open_, date_index)
    
    resample_dates = pd.Series(resample_dates).view(np.int64).values

    return [price.values,
            high.values,
            low.values,
            open_.values,
            price.index.view(np.int64),
            price.columns.astype(str).values,
            position.values,
            position.index.view(np.int64),
            position.columns.astype(str).values,
            resample_dates
            ]

def rebase(prices, value=100):
    """
    Rebase all series to a given intial value.
    This makes comparing/plotting different series
    together easier.
    Args:
        * prices: Expects a price series
        * value (number): starting value for all series.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.div(prices.iloc[0], axis=1) * value
    return prices / prices.iloc[0] * value

def sim(position: Union[pd.DataFrame, pd.Series],
        resample:Union[str, None]=None, resample_offset:Union[str, None] = None,
        position_limit:float=1, fee_ratio:float=1.425/1000,
        tax_ratio: float=3/1000, stop_loss: Union[float, None]=None,
        take_profit: Union[float, None]=None, trail_stop: Union[float, None]=None, touched_exit: bool=False,
        retain_cost_when_rebalance: bool=False, stop_trading_next_period: bool=True, live_performance_start:Union[str, None]=None,
        mae_mfe_window:int=0, mae_mfe_window_step:int=1, fast_mode=False, data=None):


     # check type of position
    if not isinstance(position.index, pd.DatetimeIndex):
        raise TypeError("Expected the dataframe to have a DatetimeIndex")
    
    if isinstance(data, Data):
        price = data.get('price:close')
    else:
        data=Data()
        price = data.get('price:close')

    high = price
    low = price
    open_ = price
    if touched_exit:
        high = data.get('price:high').reindex_like(price)
        low =data.get('price:low').reindex_like(price)
        open_ = data.get('price:open').reindex_like(price) 

    if not isinstance(price.index[0], pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)
        high.index = pd.to_datetime(high.index)
        low.index = pd.to_datetime(low.index)
        open_.index = pd.to_datetime(open_.index)

    assert len(position.shape) >= 2
    delta_time_rebalance = position.index[-1] - position.index[-3]
    backtest_to_end = position.index[-1] + \
        delta_time_rebalance > price.index[-1]

    tz = position.index.tz
    now = datetime.datetime.now(tz=tz)

    position = position[(position.index <= price.index[-1]) | (position.index <= now)]
    backtest_end_date = price.index[-1] if backtest_to_end else position.index[-1]

    # resample dates
    dates = None
    next_trading_date = position.index[-1]
    if isinstance(resample, str):

        warning_resample(resample)

        # add additional day offset
        offset_days = 0
        if '+' in resample:
            offset_days = int(resample.split('+')[-1])
            resample = resample.split('+')[0]
        if '-' in resample and resample.split('-')[-1].isdigit():
            offset_days = -int(resample.split('-')[-1])
            resample = resample.split('-')[0]

        # generate rebalance dates
        alldates = pd.date_range(
            position.index[0], 
            position.index[-1] + datetime.timedelta(days=720), 
            freq=resample, tz=tz)

        alldates += DateOffset(days=offset_days)

        if resample_offset is not None:
            alldates += to_offset(resample_offset)

        dates = [d for d in alldates if position.index[0]
                 <= d and d <= position.index[-1]]

        # calculate the latest trading date
        next_trading_date = min(
           set(alldates) - set(dates))

        if dates[-1] != position.index[-1]:
            dates += [next_trading_date]

    if stop_loss is None or stop_loss == 0:
        stop_loss = 1

    if take_profit is None or take_profit == 0:
        take_profit = np.inf

    if trail_stop is None or trail_stop == 0:
        trail_stop = np.inf

    if dates is not None:
        position = position.reindex(dates, method='ffill')

    args = arguments(price, high, low, open_, position, dates, fast_mode=fast_mode)

    creturn_value = backtest_(*args,
                              fee_ratio=fee_ratio, tax_ratio=tax_ratio,
                              stop_loss=stop_loss, take_profit=take_profit, trail_stop=trail_stop,
                              touched_exit=touched_exit, position_limit=position_limit,
                              retain_cost_when_rebalance=retain_cost_when_rebalance,
                              stop_trading_next_period=stop_trading_next_period,
                              mae_mfe_window=mae_mfe_window, mae_mfe_window_step=mae_mfe_window_step)
    
    total_weight = position.abs().sum(axis=1)

    position = position.div(total_weight.where(total_weight!=0, np.nan), axis=0).fillna(0)\
                       .clip(-abs(position_limit), abs(position_limit))
    
    creturn_dates = dates if dates and fast_mode else price.index

    creturn = (pd.Series(creturn_value, creturn_dates)
                # remove the begining of creturn since there is no pct change
                .pipe(lambda df: df[(df != 1).cumsum().shift(-1, fill_value=1) != 0])
                # remove the tail of creturn for verification
                .loc[:backtest_end_date]
                # replace creturn to 1 if creturn is None
                .pipe(lambda df: df if len(df) != 0 else pd.Series(1, position.index)))
    
    position = position.loc[creturn.index[0]:]

    daily_creturn = rebase(creturn.resample('1d').last().dropna().ffill())
    
    stock_data = pd.DataFrame(index = creturn.index)
    stock_data['portfolio_returns'] = daily_creturn
    stock_data['cum_returns'] = creturn
    stock_data['company_count'] = (position != 0).sum(axis=1)

    r = report.Report(stock_data, position)
    return r
    """
    r = report.Report(
        creturn=creturn,
        position=position,
        fee_ratio=fee_ratio,
        tax_ratio=tax_ratio,
        trade_at=trade_at_price,
        next_trading_date=next_trading_date,
        market_info=market)


    stock_data = pd.DataFrame(index=stock_price.index)
    shares_df = shares_df.reindex(stock_price.index, method="ffill")
    # stock_data['portfolio_value'] = assets['portfolio_value'].asfreq('D', method='ffill')
    stock_data['portfolio_value'] = (stock_price * shares_df).sum(axis=1)
    start_trading_day = stock_data.loc[stock_data['portfolio_value'] != 0.0].index[0]
    stock_data = stock_data.loc[start_trading_day:]

    # daily return
    stock_data['portfolio_returns'] = stock_data['portfolio_value'].pct_change(1)
    stock_data = stock_data.fillna(0).replace([np.inf, -np.inf, -1], 0)

    # 累計報酬
    stock_data['cum_returns'] = stock_data['portfolio_returns'].add(1).cumprod().sub(1)

    # 每日入選股票數量
    stock_data['company_count'] = (position != 0).sum(axis=1)
    stock_data['company_count'] = stock_data['company_count'].fillna(0)

    # 每日入選股票數量
    stock_data['company_count'] = (position != 0).sum(axis=1)
    stock_data['company_count'] = stock_data['company_count'].fillna(method='ffill')
    
    r = report.Report(stock_data, position, assets)
    return r
    """
    return position,dates,creturn,next_trading_date