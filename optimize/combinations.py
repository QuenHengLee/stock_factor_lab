from itertools import combinations
import pandas as pd
from backtest import sim
from report import Report
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

    """

    key_dataset = []
    conditions.pop('__builtins__', None)
    new_conditions = {}
    for k, v in conditions.items():
        v = CustomDataFrame(v)
        # if isinstance(v.index[0], str):
        #     v = v.index_str_to_date()
        new_conditions[k] = v
    for i in range(1, len(new_conditions) + 1):
        key_dataset.extend(list(combinations(new_conditions.keys(), i)))
    conditions_combinations = [' & '.join(k) for k in key_dataset]

    reports = {}
    for k in conditions_combinations:
        if hold_until:
            # position = eval(k, new_conditions).hold_until(**hold_until)
            position = eval(k, new_conditions)
        else:
            position = eval(k, new_conditions)

        reports[k] = sim(position, *args, **kwargs)

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
        reports = reports
        dataset = {k: v for k, v in sorted(reports.items(), key=lambda item: item[1].stock_data['cum_returns'].iloc[-1], reverse=True)}
        for k, v in dataset.items():
            series = v.stock_data['cum_returns']
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=k, meta=k,
                                        hovertemplate="%{meta}<br>Date:%{x}<br>Creturns:%{y}<extra></extra>"))
        fig.update_layout(title={'text': 'Cumulative returns', 'x': 0.49, 'y': 0.9, 'xanchor': 'center',
                                    'yanchor': 'top'})
        return fig
