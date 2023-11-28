from get_data import Data
from database import Database
from backtest import *
from datetime import datetime
import pandas as pd
from dataframe import CustomDataFrame
import pandas as pd
import numpy as np


def cal_interpolated_of_df(df, ascending=False):
    """
    INPUTS:
        df: 想要計算內插值的資料dataframe
        ascending: 決定因子是越大(F)/小(T)越好, 因子的排序方向
    RETURN:
        interpolated_df: 計算完因子內插值後的dataframe

    FUNCTION:
        以ROW為基準，計算每一ROW的內插值，最大1最小0
    """
    # 計算每行的最大值和最小值
    max_values = df.max(axis=1)
    min_values = df.min(axis=1)

    # 計算內插值
    interpolated_df = (df.sub(min_values, axis=0)).div(
        (max_values - min_values), axis=0
    )

    # 根據因子的ascending做進一步處理
    if ascending:
        return 1 - interpolated_df
    else:
        return interpolated_df


def cal_factor_sum_df_interpolated(
    factor_df_dict, factor_ratio_dict, factor_asc_dict, quantile=4
):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        factor_ratio_dict: 一個字典，包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        factor_sum_df_interpolated: 雙因子內插值相加後的加權總分
    FUNCTION:
        該因子選股的方法是根據台股研究室的內插法
        計算多個因子內插值的加權總分，如果有任一因子為nan，其他因子不為nan，則加總也是nan
        最後根據因子切割的大小quantile，回傳該權重的position
    """
    # 確保輸入的因子數量和比重數量相等
    if len(factor_df_dict) != len(factor_ratio_dict):
        raise ValueError("因子數量和比重數量不相等")

    # 計算因子DF的內插值
    factor_df_interpolated = {
        name: cal_interpolated_of_df(df, factor_asc_dict.get(name, False))
        for name, df in factor_df_dict.items()
    }

    # 將每個因子的內插值乘上對應的比重
    factor_interpolated_weighted = {
        name: interpolated * factor_ratio_dict[name]
        for name, interpolated in factor_df_interpolated.items()
    }

    # 將所有因子的加權內插值相加，得加權總分
    # 並轉成CustomDataFrame
    factor_sum_df_interpolated = sum(factor_interpolated_weighted.values())
    factor_sum_df_interpolated = CustomDataFrame(factor_sum_df_interpolated)

    # 回傳多因子權重加總後的dataframe
    return factor_sum_df_interpolated.divide_slice(quantile)


def factor_analysis_multi_ratio(factor_df_dict, factor_asc_dict, quantile=4):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        factor_sum_df_interpolated_dict: 各種因子權重組合的內插值相加後的加權總分，回傳一個包含多個df的dict
    FUNCTION:
        該因子選股的方法是根據台股研究室的內插法
        希望能夠比較不同因子權重的組合表現績效差異
    """
    # 回測以下組組合的
    muiti_ratio_dict = {
        {
            "factor1": 0,
            "factor2": 1,
        },
        {
            "factor1": 0.1,
            "factor2": 0.9,
        },
        {
            "factor1": 0.2,
            "factor2": 0.8,
        },
        {
            "factor1": 0.3,
            "factor2": 0.7,
        },
        {
            "factor1": 0.4,
            "factor2": 0.6,
        },
        {
            "factor1": 0.5,
            "factor2": 0.5,
        },
        {
            "factor1": 0.6,
            "factor2": 0.4,
        },
        {
            "factor1": 0.7,
            "factor2": 0.3,
        },
        {
            "factor1": 0.8,
            "factor2": 0.2,
        },
        {
            "factor1": 0.9,
            "factor2": 0.1,
        },
        {
            "factor1": 1,
            "factor2": 0,
        },
    }

    # 每種權重組合都帶入計算
    for ratio in muiti_ratio_dict:
        cal_factor_sum_df_interpolated(factor_df_dict, ratio, quantile)

    # TODO...


# 進行遮罩  5 MASK TRUE = 5,  3 MASK FALSE = NAN
# Achieving Alpha雙因子會用到
def MASK(df_bool, df_numeric):
    # 使用 np.where 進行遮罩操作
    result = np.where(df_bool, df_numeric, np.nan)

    # 將結果添加到新的 DataFrame 中，並設定相同的日期索引
    result_df = pd.DataFrame(result, columns=df_numeric.columns, index=df_numeric.index)

    return CustomDataFrame(result_df)


def factor_analysis_two_factor_AA(factor_df_dict, factor_asc_dict, quantile=4):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        factor_ratio_dict: 一個字典，包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        各分位的position，回傳一個包含多個df的dict
    FUNCTION:
        實現Achieving Alpha的雙因子選股方法，強調第一個因子，弱化第二個因子

    """
    # 取得因子的list
    # 取得所有的鍵
    keys = factor_df_dict.keys()
    # 將鍵轉換為列表（可選）
    factor_list = list(keys)
    # 取得個因子名稱
    factor_1 = factor_list[0]
    factor_2 = factor_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = factor_df_dict[factor_1]
    factor_2_df = factor_df_dict[factor_2]
    # 從Input擷取個因子的排序方向
    factor_1_asc = factor_asc_dict[factor_1]
    factor_2_asc = factor_asc_dict[factor_2]
    # 先將第一個因子根據quantile值做切割
    factor_1_slice = CustomDataFrame(factor_1_df).divide_slice(quantile, factor_1_asc)
    # 先進行MASK處理
    factor1_mask_factor2 = {}
    for q, df in factor_1_slice.items():
        # key = 'Quantile_1_MASK_factor2'
        key = f"{q}_MASK_factor2"
        value = MASK(df, factor_2_df)
        factor1_mask_factor2[key] = value

    # print(factor1_mask_factor2)

    result = {}
    # print("因子:", factor_list)
    for i in range(quantile):
        # key = f"Quantile{i+1}_{factor_1}_{factor_2}"
        key = f"Quantile_{i+1}"
        tmp_str = "Quantile_" + str(i + 1) + "_MASK_factor2"
        tmp_list = factor1_mask_factor2[tmp_str].divide_slice(quantile, factor_2_asc)
        result[key] = tmp_list["Quantile_" + str(i + 1)]
    print(result)
    return result


def factor_analysis_two_factor(factor_df_dict, factor_asc_dict, quantile=4):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        factor_ratio_dict: 一個字典，包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        quantile: 打算將因子切割成幾等分
    RETURN:
        各分位的position，回傳一個包含多個df的dict
    FUNCTION:
        將兩個因子DF經divide_slice後，根據Quantile 執行AND運算
    """
    # 取得因子的list
    # 取得所有的鍵
    keys = factor_df_dict.keys()
    # 將鍵轉換為列表（可選）
    factor_list = list(keys)
    # 取得個因子名稱
    factor_1 = factor_list[0]
    factor_2 = factor_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = factor_df_dict[factor_1]
    factor_2_df = factor_df_dict[factor_2]
    # 從Input擷取個因子的排序方向
    factor_1_asc = factor_asc_dict[factor_1]
    factor_2_asc = factor_asc_dict[factor_2]

    factor_1_after_slice = factor_1_df.divide_slice(quantile, factor_1_asc)
    factor_2_after_slice = factor_2_df.divide_slice(quantile, factor_2_asc)
    # print(factor_1_after_slice)
    result = {}
    for i in range(quantile):
        # key = f"Quantile{i+1}_{factor_1}_{factor_2}"
        key = f"Quantile_{i+1}"
        value = (
            factor_1_after_slice["Quantile_" + str(i + 1)]
            & factor_2_after_slice["Quantile_" + str(i + 1)]
        )
        result[key] = value

    return result


def factor_analysis_single(factor_df_dict, factor_asc_dict, quantile=4):
    """
    INPUTS:
        factor_df_dict: 一個字典，包含多個因子的dataframe，以因子名稱為鍵，對應的dataframe為值
        factor_asc_dict: 一個字典，包含多個因子的排序方向，越大/小越好
        quantile: 打算將因子切割成幾等分
    RETURN:
        各分位的position，回傳一個包含多個df的dict
    FUNCTION:
        將兩個因子DF經divide_slice後，根據Quantile 執行AND運算
    """
    if len(factor_df_dict) == 1 and len(factor_asc_dict) == 1:
        # 取得所有的鍵
        keys = factor_df_dict.keys()
        # 將鍵轉換為列表（可選）
        factor_name_list = list(keys)
        # 取得個因子名稱
        factor_name = factor_name_list[0]
        factor_value_df = factor_df_dict[factor_name]
        factor_asc = factor_asc_dict[factor_name]
        return factor_value_df.divide_slice(quantile, factor_asc)
    else:
        print("該方法為單因子切割，請勿帶入超過一個因子")


if __name__ == "__main__":
    # 生成 10x10 的隨機數據，其中大約 20% 的元素為 NaN
    np.random.seed(42)
    data = np.random.randint(0, 1000, size=(100, 100)).astype(float)
    mask = np.random.choice([True, False], size=data.shape, p=[0.2, 0.8])
    data[mask] = np.nan
    # 將數據轉換成 DataFrame，並使用日期範圍作為索引
    factor_1 = pd.DataFrame(
        data,
        columns=[f"Col{i+1}" for i in range(100)],
        index=pd.date_range(start="2022-01-01", periods=100, freq="D"),
    )

    # 生成 10x10 的隨機數據，其中大約 20% 的元素為 NaN
    np.random.seed(98)
    data = np.random.randint(0, 1000, size=(100, 100)).astype(float)
    mask = np.random.choice([True, False], size=data.shape, p=[0.2, 0.8])
    data[mask] = np.nan
    # 將數據轉換成 DataFrame，並使用日期範圍作為索引
    factor_2 = pd.DataFrame(
        data,
        columns=[f"Col{i+1}" for i in range(100)],
        index=pd.date_range(start="2022-01-01", periods=100, freq="D"),
    )

    # 示例使用方式
    factor_df_dict = {
        "ROE": CustomDataFrame(factor_1),
        "PB": CustomDataFrame(factor_2),
        # 可以加入其他因子
    }

    factor_ratio_dict = {
        "ROE": 0.5,
        "PB": 0.5,
        # 可以加入其他比重
    }
    factor_asc_dict = {
        "ROE": False,
        "PB": False,
    }
    # 呼叫台股研究室的內積加權法cal_factor_sum_df_interpolated()
    # result = cal_factor_sum_df_interpolated(
    #     factor_df_dict, factor_ratio_dict, factor_asc_dict
    # )

    # 呼叫Achieving Alpha的雙因子選股方法factor_analysis_two_factor_AA()
    result = factor_analysis_two_factor_AA(factor_df_dict, factor_asc_dict)

    # # 呼叫直接兩因子做AND運算方法
    # result = factor_analysis_two_factor(factor_df_dict, factor_asc_dict)
