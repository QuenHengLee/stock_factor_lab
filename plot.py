import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


class Plot:
    def __init__(self):
        pass

    def plot(
        self,
        backtest_result,
        index="portfolio_value",
    ):
        """
        INPUT: backtest_result各quantile的回測結果, index要畫的指標
        OUTPUT: 折線圖表
        FUNCTION: 畫出各個quantile的回測結果
        """
        size = len(backtest_result)
        # 假設有N組quantile，將它們存儲在一個名為 quantile_values 的列表中
        series_list = []
        size = len(backtest_result)
        for i in range(1, size + 1):
            tmp = backtest_result[f"Quantile {i}"]["stock_data"][index]
            series_list.append(tmp)

        # 使用pd.concat()将Series合并成一个DataFrame
        df = pd.concat(series_list, axis=1)

        # 如果需要，你可以为DataFrame的列指定列名
        # 例如，你可以使用以下方式为每列命名
        # column_names = ['Quantile 1', 'Quantile 2', 'Quantile 3', 'Quantile 4']
        column_names = [f"Quantile {i}" for i in range(1, size + 1)]
        df.columns = column_names

        # 设置Seaborn的风格
        # sns.set(style="darkgrid")

        # # 创建一个绘图
        # plt.figure(figsize=(10, 6))  # 设置图形的大小

        # # 使用Seaborn的lineplot函数绘制N个折线图
        # # 将每个Series的数据列名传递给x和y参数，并使用hue参数指定颜色分组
        # sns.lineplot(data=df, markers=True, dashes=False)  # 可以根据需要自定义其他参数

        # # 添加标题和标签
        # plt.title("Quantile Backtest Result")
        # plt.xlabel("Time Period")
        # plt.ylabel(index)

        # # 显示图例
        # plt.legend(loc="best")

        # # 显示图形
        # plt.show()

        # 另一種動態互動套件畫圖
        # 使用Plotly Express繪製折線圖
        fig = px.line(df, x=df.index, y=df.columns, title=index + "折線圖")
        # 顯示圖表
        fig.show()

    def table(
        self,
        backtest_result,
        index="portfolio_value",
        N=0,
    ):
        """
        INPUT: backtest_result各quantile的回測結果, index要畫的指標, quantile要畫的第幾個quantile
        OUTPUT: Dataframe
        FUNCTION: 畫出各個quantile的回測結果Dataframe, N=0代表全部quantile
        """
        size = len(backtest_result)
        # 假設有N組quantile，將它們存儲在一個名為 quantile_values 的列表中
        series_list = []
        size = len(backtest_result)
        for i in range(1, size + 1):
            tmp = backtest_result[f"Quantile {i}"]["stock_data"][index]
            series_list.append(tmp)

        # 使用pd.concat()将Series合并成一个DataFrame
        df = pd.concat(series_list, axis=1)

        # 如果需要，你可以为DataFrame的列指定列名
        # 例如，你可以使用以下方式为每列命名
        # column_names = ['Quantile 1', 'Quantile 2', 'Quantile 3', 'Quantile 4']
        column_names = [f"Quantile {i}" for i in range(1, size + 1)]
        df.columns = column_names

        if N == 0:
            return df
        else:
            return df[f"Quantile {N}"].to_frame()
