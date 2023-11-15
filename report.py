import pandas as pd
import numpy as np
import plotly.express as px


# 用來安全進行除法的函數。如果分母 d 不等於零，則返回 n / d，否則返回 0。
def safe_division(n, d):
    return n / d if d else 0

# 用來將dataframe按照月份排列
def sort_month(df):
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return df[month_order]

class Report():
    def __init__(self, stock_data) -> None:
        self.stock_data = stock_data

    def display(self):
        from IPython.display import display
        
        # 計算
        drawdown, mdd = self.calc_mdd()
        imp_stats = pd.Series({
         'annualized_rate_of_return':str(round(self.calc_cagr()*100, 2))+'%',
         'sharpe': str(0),
         'max_drawdown':str(round(mdd*100, 2))+'%',
         'win_ratio':str(round(self.calc_win_ratio()*100, 2))+'%',
        }).to_frame().T
        imp_stats.index = ['']

        yearly_return_fig = self.create_yearly_return_figure()
        monthly_return_fig = self.create_monthly_return_figure()

        # show出來
        display(imp_stats)
        display(yearly_return_fig)
        display(monthly_return_fig)
    
    def create_monthly_return_figure(self):
        import plotly.express as px
        # 計算月回報，並儲存在一個df
        monthly_table = self.calc_monthly_return()

        fig = px.imshow(monthly_table.values,
                        labels=dict(x="month", y='year', color="return(%)"),
                        x=monthly_table.columns.astype(str),
                        y=monthly_table.index.astype(str),
                        text_auto=True,
                        color_continuous_scale='RdBu_r',

                        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "year: %{y}",
                "month: %{x}",
                "return: %{z}%",
            ])
        )

        fig.update_layout(
            height = 550,
            width= 800,
            margin=dict(l=20, r=270, t=40, b=40),
            title={
                'text': 'monthly return',
                'x': 0.025,
                'yanchor': 'top',
            },
            yaxis={
                'side': "right",
            },
            coloraxis_showscale=False,
            coloraxis={'cmid':0}
        )

        return fig
    
    def create_yearly_return_figure(self):
        import plotly.express as px
        yearly_return = self.calc_yearly_return()

        fig = px.imshow(yearly_return.values,
                        labels=dict(color="return(%)"),
                        x=yearly_return.columns.astype(str),
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        )

        fig.update_traces(
            hovertemplate="<br>".join([
                "year: %{x}",
                "return: %{z}%",
            ])
        )

        fig.update_layout(
            height = 120,
            width= 800,
            margin=dict(l=20, r=270, t=40, b=40),
            yaxis={
                'visible': False,
            },
            title={
                'text': 'yearly return',
                'x': 0.025,
                'yanchor': 'top',
            },
            coloraxis_showscale=False,
            coloraxis={'cmid':0}
            )

        return fig

    def calc_win_ratio(self):
        '''
        計算勝率是看每天報酬>0的天數/總天數
        '''
        trades = self.stock_data.replace([0],np.nan).dropna()
        return sum(trades['portfolio_returns'] > 0) / len(trades) if len(trades) != 0 else 0

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
        # end = dd.idxmin()
        # start = r.loc[:end].idxmax()
        # days = end-start

        return dd, mdd

    def calc_cagr(self):
        '''
        計算CAGR用 (最終價值/初始價值) ^ (1/持有時間(年)) - 1
        那其實 最終價值/初始價值 跟「累計回報」會差不多，
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