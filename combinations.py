from itertools import combinations
import pandas as pd
from backtest import sim
from report import Report
from dataframe import CustomDataFrame
from tqdm import tqdm

def sim_conditions(conditions, combination=False, *args, **kwargs):
    """取得回測報告集合

    將選股條件排出所有的組合並進行回測，方便找出最好條件的交集結果。

    Args:
      conditions (dict): 選股條件集合，key 為條件名稱，value 為條件變數，ex:`{'c1':c1, 'c2':c2}`
      hold_until (dict): 設定[訊號進出場語法糖](https://doc.finlab.tw/reference/dataframe/#finlab.dataframe.FinlabDataFrame.hold_until)參數，預設為不使用。ex:`{'exit':exit, 'stop_loss':0.1}`
      combination (bool): 是否要排列組合。
      *args (tuple): `finlab.backtest.sim()` 回測參數設定。
      **kwargs (dict): `finlab.backtest.sim()` 回測參數設定。

    Returns:
      (finlab.optimize.combination.ReportCollection):回測數據報告

    """

    reports = {}
    for k, v in tqdm(conditions.items(), desc="Backtesting progress", unit="condition"):
        reports[k] = sim(v, *args, **kwargs)
    for k, v in tqdm(conditions.items(), desc="Backtesting progress", unit="condition"):
        reports[k] = sim(v, *args, **kwargs)

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
        self.stats = self.get_stats()
        self.stats = self.get_stats()

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
                reports.items(),
                key=lambda item: item[1].stock_data["cum_returns"].iloc[-1],
                reverse=True,
            )
        }
        for k, v in dataset.items():
            series = v.stock_data["cum_returns"]
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

    # 將欲畫出的回測績效指標以參數方式帶入method
    def plot_custom_metric(self, metric_column="portfolio_value"):
        """繪製策略自定義指標的變化曲線

        比較策略自定義指標曲線變化

        Args:
            metric_column (str): 要繪製的指標數據列名稱，預設為 'portfolio_value'，
            亦可輸入: portfolio_value, portfolio_returns	, cum_returns, company_count

        Returns:
            (plotly.graph_objects.Figure): 折線圖

        Examples:
            ![line](img/optimize/report_collection_custom_metric.png)
        """
        import plotly.graph_objects as go

        # 建立一個空的 Figure
        fig = go.Figure()

        # 從類別實例中取得報告
        reports = self.reports

        # 根據最後一個數據點的指標值對報告進行排序
        dataset = {
            k: v
            for k, v in sorted(
                reports.items(),
                key=lambda item: item[1].stock_data[metric_column].iloc[-1],
                reverse=True,
            )
        }

        # 迭代排序後的數據集，將跡線添加到圖中
        for k, v in dataset.items():
            series = v.stock_data[metric_column]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=k,
                    meta=series.index,  # Use series index as meta information (you can adjust this based on your data)
                    hovertemplate="%{meta}<br>Date:%{x}<br>metric_value: %{y}<extra></extra>",
                )
            )

        # 更新圖的佈局，包括標題
        fig.update_layout(
            title={
                "text": "{} 曲線".format(metric_column),
                "x": 0.49,
                "y": 0.9,
                "xanchor": "center",
                "yanchor": "top",
                # Add other layout configurations as needed
            }
        )

        # 顯示圖表
        # fig.show()
        return fig

    def get_stats(self):
        """取得策略指標比較表

        指標欄位說明：

        * `'CAGR'`: 策略年化報酬
        * `'CAGR'`: 策略年化報酬
        * `'daily_sharpe'`: 策略年化夏普率
        * `'max_drawdown'`: 策略報酬率最大回撤率(負向)
        * `'avg_drawdown'`: 策略平均回撤(負向)
        * `'ytd'`: 今年度策略報酬率
        * `'win_ratio'`: 每筆交易勝率

        Returns:
          (pd.DataFrame): 策略指標比較報表
        """

        def get_strategy_indicators(report):
            if isinstance(report, Report):
                stats = report.get_stats()
                strategy_indexes = {n: stats[n] for n in
                                    ['CAGR', 'daily_sharpe',
                                     'max_drawdown', 'avg_drawdown', 
                                     'win_ratio', 'ytd']}
                strategy_indexes = {
                    n: stats[n]
                    for n in [
                        "CAGR",
                        "daily_sharpe",
                        "max_drawdown",
                        "avg_drawdown",
                        "win_ratio",
                        "ytd",
                    ]
                }
                # trade_indexes.update(
                #     {f'avg_{n}': trades[n].mean() for n in ['return', 'mae', 'bmfe', 'gmfe', 'mdd']})
                # strategy_indexes.update(trade_indexes)
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

        """
        if self.stats is None:
            self.get_stats()
        df = self.stats

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
                df.T.sort_values("CAGR", ascending=False)
                .style.set_caption("Backtest combinations heatmap")
                .background_gradient(axis=0, cmap="YlGn")
            )
