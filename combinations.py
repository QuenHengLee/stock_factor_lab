from itertools import combinations
import pandas as pd

# from finlab.backtest import sim

from backtest import *
from finlab.core.report import Report
from finlab.utils import logger
from dataframe import CustomDataFrame


def sim_conditions(conditions, hold_until={}, *args, **kwargs):
    """取得回測報告集合

    將選股條件排出所有的組合並進行回測，方便找出最好條件的交集結果。

    Args:
      conditions (dict): 選股條件集合，key 為條件名稱，value 為條件變數，ex:`{'c1':c1, 'c2':c2}`
      hold_until (dict): 設定[訊號進出場語法糖](https://doc.finlab.tw/reference/dataframe/#finlab.dataframe.FinlabDataFrame.hold_until)參數，預設為不使用。ex:`{'exit':exit, 'stop_loss':0.1}`
      *args (tuple): `finlab.backtest.sim()` 回測參數設定。
      **kwargs (dict): `finlab.backtest.sim()` 回測參數設定。

    Returns:
      (finlab.optimize.combination.ReportCollection):回測數據報告

    Examples:
        ```py
        from finlab import data
        from finlab.backtest import sim
        from finlab.optimize.combinations import sim_conditions

        close = data.get("price:收盤價")
        pe = data.get('price_earning_ratio:本益比')
        rev=data.get('monthly_revenue:當月營收').index_str_to_date()
        rev_ma3=rev.average(3)
        rev_ma12=rev.average(12)

        c1 = (close > close.average(20)) & (close > close.average(60))
        c2 = (close == close.rolling(20).max())
        c3 = pe < 15
        c4 = rev_ma3/rev_ma12 > 1.1
        c5 = rev/rev.shift(1) > 0.9
        exits = close < close.average(20)
        conditions = {'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4, 'c5':c5}
        report_collection = sim_conditions(conditions=conditions, hold_until={'exit':exits, 'stop_loss':0.1}, resample='M', position_limit=0.1, upload=False)
        # 策略分組指標報告
        # print(report_collection.stats)
        report_collection.plot_creturns().show()
        report_collection.plot_stats('bar').show()
        report_collection.plot_stats('heatmap')
        ```
        視覺化範例

        ex1:

        `report_collection.plot_creturns().show()`

        繪製折線圖指標分群棒狀圖

        ![bar](img/optimize/report_collection_creturns.png)

        ex2:

        `report_collection.plot_stats('bar').show()`

        繪製指標分群棒狀圖

        ![bar](img/optimize/report_collection_stats_bar.png)

        ex3:

        `report_collection.plot_stats('heatmap')`

        繪製指標分級熱力圖，數值越大為排名越前面，avg_score為指標平均分數，分數越高為評價較正向的策略。

        ![heatmap](img/optimize/report_collection_stats_heatmap.png)
    """

    key_dataset = []
    conditions.pop("__builtins__", None)
    new_conditions = {}
    # 將conditions裡面的資料經過處理(轉DF、轉日期)後，存放在new_conditions
    # new_conditions = {'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4, 'c5':c5}
    for k, v in conditions.items():
        try:
            v = CustomDataFrame(v)
            if isinstance(v.index[0], str):
                v = v.index_str_to_date()
            new_conditions[k] = v
        except:
            logger.error(f"{k} data formaat must be dataframe.")
    # 這邊是實際做排列組合的運算
    for i in range(1, len(new_conditions) + 1):
        key_dataset.extend(list(combinations(new_conditions.keys(), i)))
    # conditions_combinations會根據conditions存放各種可能的排列組合
    conditions_combinations = [" & ".join(k) for k in key_dataset]

    reports = {}
    # 將可能組合一個一個進行回測sim()
    for k in conditions_combinations:
        # 若有出場條件...
        if hold_until:
            position = eval(k, new_conditions).hold_until(**hold_until)
        else:
            position = eval(k, new_conditions)
        # 進行回測，結果存入reports
        try:
            # 這邊的sim()可以跟hao的接上去
            reports[k] = Backtest.sim(position, *args, **kwargs)
        except Exception as error:
            logger.error(f"{k}:{error}")
            pass

    return ReportCollection(reports)


class ReportCollection:
    def __init__(self, reports):
        """回測組合比較報告

        判斷策略組合數據優劣，從策略海中快速找到體質最強的策略。
        也可以觀察在同條件下的策略疊加更多條件後會有什麼變化？
        Args:
          reports (dict): 回測物件集合，ex:`{'strategy1': finlab.backtest.sim(),'strategy2': finlab.backtest.sim()}`
        """
        self.reports = reports
        self.stats = None

    def get_stats(self):
        """取得策略指標比較表

        指標欄位說明：

        * `'daily_mean'`: 策略年化報酬
        * `'daily_sharpe'`: 策略年化夏普率
        * `'daily_sortino'`: 策略年化索提諾比率
        * `'max_drawdown'`: 策略報酬率最大回撤率(負向)
        * `'avg_drawdown'`: 策略平均回撤(負向)
        * `'ytd'`: 今年度策略報酬率
        * `'win_ratio'`: 每筆交易勝率
        * `'avg_return'`: 每筆交易平均獲利率
        * `'avg_mae'`: 每筆交易平均最大不利方向幅度(負向)
        * `'avg_bmfe'`: 最大不利方向發生前的"每筆交易平均最大有利方向幅度"，若數值越高，越有機會在停損之前操作停利。
        * `'avg_gmfe'`: 每筆交易平均最大有利方向幅度
        * `'avg_mdd'`: 每筆交易平均的最大回撤率(負向)

        Returns:
          (pd.DataFrame): 策略指標比較報表
        """

        def get_strategy_indicators(report):
            if isinstance(report, Report):
                stats = report.get_stats()
                trades = report.trades
                strategy_indexes = {
                    n: stats[n]
                    for n in [
                        "daily_mean",
                        "daily_sharpe",
                        "daily_sortino",
                        "max_drawdown",
                        "avg_drawdown",
                    ]
                }
                trade_indexes = {"win_ratio": stats["win_ratio"]}
                trade_indexes.update(
                    {
                        f"avg_{n}": trades[n].mean()
                        for n in ["return", "mae", "bmfe", "gmfe", "mdd"]
                    }
                )
                strategy_indexes.update(trade_indexes)
                return strategy_indexes

        df = pd.DataFrame(
            {k: get_strategy_indicators(v) for k, v in self.reports.items()}
        )
        self.stats = df
        return df

    def plot_stats(self, mode="bar", heatmap_sort_by="avg_score", indicators=[]):
        """策略指標比較報表視覺化

        Args:
          mode (str): 繪圖模式。`'bar'` - 指標分群棒狀圖。`'heatmap'` - 指標分級熱力圖。
          heatmap_sort_by (str or list of str): heatmap 降冪排序的決定欄位
          indicators (list): 要顯示的特定指標欄位，預設為將指標全部顯示

        Returns:
          (plotly.graph_objects.Figure): 長條圖
          (pd.DataFrame): 熱力圖

        Examples:
            ex1:

            繪製指標分群棒狀圖

            ![bar](img/optimize/report_collection_stats_bar.png)

            ex2:

            繪製指標分級熱力圖。

            `'avg_score'`: 各指標加總後的平均分數，分數越高為整體評價較正向的策略。

            預設以avg_score為排序，數值越大為排名越前面，分數越高為整體評價較優的策略。

            ![heatmap](img/optimize/report_collection_stats_heatmap.png)
        """
        if self.stats is None:
            self.get_stats()
        df = self.stats
        if len(indicators) > 0:
            try:
                df = df.loc[indicators]
            except KeyError:
                logger.error(f"Indicators selection must be in {list(df.index)}")
        if mode == "bar":
            import plotly.graph_objects as go

            items = df.columns
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df.index,
                        y=df[item],
                        name=item,
                        meta=[item],
                        hovertemplate="%{meta}<br>%{label}<br>%{y}<extra></extra>",
                    )
                    for item in items
                ]
            )
            # Change the bar mode
            fig.update_layout(
                title={
                    "text": "Backtest combinations stats",
                    "x": 0.49,
                    "y": 0.9,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                barmode="group",
            )
            return fig

        elif mode == "heatmap":
            return (
                df.rank(pct=True, axis=1)
                .transpose()
                .assign(avg_score=lambda d: d.mean(axis=1))
                .round(2)
                .mul(100)
                .sort_values(heatmap_sort_by, ascending=False)
                .style.set_caption("Backtest combinations heatmap")
                .format("{:.1f}%")
                .background_gradient(axis=None, vmin=0, vmax=100, cmap="plasma")
            )

    def plot_creturns(self):
        """繪製策略累積報酬率

        比較策略淨值曲線變化

        Returns:
          (plotly.graph_objects.Figure): 折線圖

        Examples:
            ![line](img/optimize/report_collection_creturns.png)
        """
        import plotly.graph_objects as go

        fig = go.Figure()
        reports = self.reports
        dataset = {
            k: v
            for k, v in sorted(
                reports.items(), key=lambda item: item[1].creturn[-1], reverse=True
            )
        }
        for k, v in dataset.items():
            series = v.creturn
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=k,
                    meta=k,
                    hovertemplate="%{meta}<br>Date:%{x}<br>Creturns:%{y}<extra></extra>",
                )
            )
        fig.update_layout(
            title={
                "text": "Cumulative returns",
                "x": 0.49,
                "y": 0.9,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        return fig
