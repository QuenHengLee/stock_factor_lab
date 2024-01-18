from get_data import Data
from database import Database
from backtest import *
from datetime import datetime
import pandas as pd
from dataframe import CustomDataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from operations import *


import numpy as np


# 雙因子切割(內插加權法用 - 固定單一權重)
def cal_factor_sum_df_interpolated(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
):
    """
    Args:
        factor_name_list (list): 包含多個因子的名稱，例如: factor_name_list = ['roe','pb']
        factor_ratio_dict (dict): 包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        factor_asc_dict (dict): 一個字典，包含多個因子的排序方向
        quantile (positive-int): 打算將因子切割成幾等分
        all_factor_df_dict (dict): 將所有因子資料的DF存在一個dict當中，例如: all_factor_df_dict = {'roe': roe_data_df, 'pb': pb_data_df, ...}
    Returns:
        factor_sum_df_interpolated (dict): 雙因子內插值相加後的加權總分
    Function:
        該因子選股的方法是根據台股研究室的內插法
        計算多個因子內插值的加權總分，如果有任一因子為nan，其他因子不為nan，則加總也是nan
        最後根據因子切割的大小quantile，回傳該權重的position
    """
    # 取得個因子名稱
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    factor_df_dict = {}
    # 判斷雙因子是否相同
    if factor_1 == factor_2:
        factor_df_dict[factor_1] = all_factor_df_dict[factor_1]
        # 將第二個因子KEY值做出名稱差異
        factor_2 = factor_2 + "'"
        factor_df_dict[factor_2] = all_factor_df_dict[factor_1]
        # 比重、排序也要加上第二個重複因子的值\
        factor_ratio_dict[factor_2] = factor_ratio_dict[factor_1]
        factor_asc_dict[factor_2] = factor_asc_dict[factor_1]
    else:
        factor_df_dict[factor_1] = all_factor_df_dict[factor_1]
        factor_df_dict[factor_2] = all_factor_df_dict[factor_2]

    # 計算因子DF的內插值
    # 初始化一個空字典以存儲插值後的數據框
    factor_df_interpolated = {}
    # 遍歷 factor_df_dict 中的每一個鍵值對
    for name, df in factor_df_dict.items():
        # 從 factor_asc_dict 中取得相應的因子，如果未找到則默認為 False
        factor = factor_asc_dict.get(name, False)
        # 調用 cal_interpolated_of_df 函數，傳入當前的數據框和因子
        interpolated_df = cal_interpolated_of_df(df, factor)
        # 將結果添加到 factor_df_interpolated 字典中
        factor_df_interpolated[name] = interpolated_df

    # factor_df_interpolated = {
    #     name: cal_interpolated_of_df(df, factor_asc_dict.get(name, False))
    #     for name, df in factor_df_dict.items()
    # }

    # 將每個因子的內插值乘上對應的比重
    factor_interpolated_weighted = {
        name: interpolated * factor_ratio_dict[name]
        for name, interpolated in factor_df_interpolated.items()
    }

    # 將所有因子的加權內插值相加，得加權總分，並轉成CustomDataFrame
    factor_sum_df_interpolated = CustomDataFrame(
        sum(factor_interpolated_weighted.values())
    )

    # 回傳多因子權重加總後的dataframe
    return factor_sum_df_interpolated.divide_slice(quantile)


# 雙因子切割(過濾篩選多因子)
def factor_analysis_two_factor_AA(
    factor_name_list: list,
    factor_asc_dict: dict,
    quantile: int = 4,
    all_factor_df_dict: dict = None,
) -> dict:
    """
    實現 Achieving Alpha 的雙因子選股方法(過濾篩選)，
    強調第一個因子，減弱第二個因子的影響。

    Args:
        factor_name_list (list): 包含多個因子名稱的列表（例如，['roe', 'pb']）。
        factor_asc_dict (dict): 包含多個因子排序方向的字典。
        quantile (positive-int): 進行因子切割的分位數。
        all_factor_df_dict (dict): 包含所有因子資料框的字典
                                  （例如，{'roe': roe_data_df, 'pb': pb_data_df, ...}）。

    Returns:
        dict: 包含每個分位數的持倉的字典。

    """

    # 取得個因子名稱()
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = CustomDataFrame(all_factor_df_dict[factor_1])
    factor_2_df = CustomDataFrame(all_factor_df_dict[factor_2])
    # 從Input擷取個因子的排序方向
    factor_1_asc = factor_asc_dict[factor_1]
    factor_2_asc = factor_asc_dict[factor_2]
    # 先將第一個因子根據quantile值做切割
    factor_1_slice_dict = factor_1_df.divide_slice(quantile, factor_1_asc)
    # 先進行MASK處理
    factor1_mask_factor2 = {}
    for q, df in factor_1_slice_dict.items():
        # key = 'Quantile_1_MASK_factor2'
        key = f"{q}_MASK_factor2"
        value = MASK(df, factor_2_df)
        factor1_mask_factor2[key] = value

    result = {}
    for i in range(quantile):
        # key = f"Quantile{i+1}_{factor_1}_{factor_2}"
        key = f"Quantile_{i+1}"
        tmp_str = "Quantile_" + str(i + 1) + "_MASK_factor2"
        tmp_list = factor1_mask_factor2[tmp_str].divide_slice(quantile, factor_2_asc)
        result[key] = tmp_list["Quantile_" + str(i + 1)]
    return result


# 雙因子切割(直接做AND交集運算)
def factor_analysis_two_factor_AND(
    factor_name_list: list,
    factor_asc_dict: dict,
    quantile: int = 4,
    all_factor_df_dict: dict = None,
) -> dict:
    """將兩個因子DF經divide_slice後，根據Quantile 執行AND運算

    Args:
        factor_name_list (list): 包含多個因子名稱的列表（例如，['roe', 'pb']）。
        factor_asc_dict (dict): 包含多個因子排序方向的字典。
        quantile (positive-int): 進行因子切割的分位數。
        all_factor_df_dict (dict): 包含所有因子資料框的字典
                                  （例如，{'roe': roe_data_df, 'pb': pb_data_df, ...}）。

    Returns:
        dict: 包含每個分位數的持倉的字典。

    """

    # 取得個因子名稱
    factor_1 = factor_name_list[0]
    factor_2 = factor_name_list[1]
    # 從Input擷取個因子的DF
    factor_1_df = all_factor_df_dict[factor_1]
    factor_2_df = all_factor_df_dict[factor_2]
    # # 從Input擷取個因子的DF
    # factor_1_df = factor_df_dict[factor_1]
    # factor_2_df = factor_df_dict[factor_2]
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


# 單因子切割
def factor_analysis_single(factor_df, factor_asc: bool, quantile: int = 4) -> dict:
    """
    單因子根據值的大小與排序方向做分割

    Args:
        factor_df (dataframe): 單一因子的資料
        factor_asc (bool): 排序的方向，F:越大越好; T:越小越好
        quantile (positive-int): 打算將因子切割成幾等分

    Returns:
        各分位的position，回傳一個包含多個df的dict
    """

    return factor_df.divide_slice(quantile, factor_asc)


# 雙因子切割(統一呼叫介面)
def factor_analysis_multi_dual(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
    method: str = "AND",
):
    """
    之後統一從這裡呼叫雙因子切割，需帶參數決定要哪種方法

    Args:
        factor_name_list (list): 包含多個因子的名稱，例如: factor_name_list = ['roe','pb']
        factor_ratio_dict (dict): 包含多個因子的比重，以因子名稱為鍵，對應的比重為值
        factor_asc_dict (dict): 一個字典，包含多個因子的排序方向
        quantile (positive-int): 打算將因子切割成幾等分
        all_factor_df_dict (dict): 將所有因子資料的DF存在一個dict當中，例如: all_factor_df_dict = {'roe': roe_data_df, 'pb': pb_data_df, ...}
        method (str): 要執行哪一種雙因子運算(AND、INTERPOLATED、AA)

    Returns:
        factor_sum_df_interpolated (dict): 雙因子內插值相加後的加權總分
    """

    if method == "AND":
        return factor_analysis_two_factor_AND(
            factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
        )
    elif method == "AA":
        return factor_analysis_two_factor_AA(
            factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
        )
    elif method == "INTERPOLATED":
        return cal_factor_sum_df_interpolated(
            factor_name_list,
            factor_ratio_dict,
            factor_asc_dict,
            quantile,
            all_factor_df_dict,
        )


# 取得三種不同雙因子方法的切割結果api
def quantile_3_diff_method(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    quantile_th: int = 1,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
) -> Dict:
    """
    取得各自方法的Qunatile_Nth，
    Args:
        factor_name_list (list of str): 打算回測哪些因子，存入陣列中
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        data (get_data.data): 傳入價量資料，以便回測使用
        quantile (positive-int): 決定因子分析要切成幾個等份
        quantile_th (positive-int): 這次報告要產出第幾分位
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
    Returns:
        (Dict[str, pd.dataframe]): 記錄個雙因子模型的Quantile_n

    """
    # 執行因子分析
    result_AA = factor_analysis_two_factor_AA(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    result_INTERPOLATED = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )
    result_AND = factor_analysis_two_factor_AND(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 選取第幾分位
    quantile_th = "Quantile_" + str(quantile_th)
    postion_AA = result_AA[quantile_th]
    postion_INTERPOLATED = result_INTERPOLATED[quantile_th]
    postion_AND = result_AND[quantile_th]
    # 建立回傳dict
    result = {}
    result["method_AA"] = postion_AA
    result["method_INTERPOLATED"] = postion_INTERPOLATED
    result["method_AND"] = postion_AND

    return result


# 取得八種不同雙因子方法的切割結果api(2個單因子+內插加權法3種不同全重+2個AA+1個AND)
def quantile_8_diff_method(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    quantile_th: int = 1,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
) -> Dict:
    """
    取得各自方法的Qunatile_Nth，
    Args:
        factor_name_list (list of str): 打算回測哪些因子，存入陣列中
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        data (get_data.data): 傳入價量資料，以便回測使用
        quantile (positive-int): 決定因子分析要切成幾個等份
        quantile_th (positive-int): 這次報告要產出第幾分位
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
    Returns:
        (Dict[str, pd.dataframe]): 記錄個雙因子模型的Quantile_n

    """
    # 不同權重的加權內插法
    f1 = factor_name_list[0]
    f2 = factor_name_list[1]
    # 執行單因子分析
    # 第一個因子
    result_f1 = factor_analysis_single(
        all_factor_df_dict[f1], factor_asc_dict[f1], quantile
    )
    # 第二個因子
    result_f2 = factor_analysis_single(
        all_factor_df_dict[f2], factor_asc_dict[f2], quantile
    )
    # 執行雙因子分析
    # 過濾篩選法(f1_f2)
    result_AA = factor_analysis_two_factor_AA(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 過濾篩選法(f2_f1)
    re_factor_name_list = factor_name_list[::-1]
    result_AA_R = factor_analysis_two_factor_AA(
        re_factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )

    # 第一種是 0.5:0.5
    result_INTERPOLATED_5_5 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )
    # 第二種全重 0.7:0.3
    factor_ratio_dict_1 = {}
    factor_ratio_dict_1[f1] = 0.7
    factor_ratio_dict_1[f2] = 0.3
    result_INTERPOLATED_7_3 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict_1,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )
    # 第三種全重 0.3:0.7
    factor_ratio_dict_2 = {}
    factor_ratio_dict_2[f1] = 0.3
    factor_ratio_dict_2[f2] = 0.7
    result_INTERPOLATED_3_7 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict_2,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )

    result_AND = factor_analysis_two_factor_AND(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 選取第幾分位
    quantile_th = "Quantile_" + str(quantile_th)
    position_f1 = result_f1[quantile_th]
    position_f2 = result_f2[quantile_th]
    postion_AA = result_AA[quantile_th]
    postion_AA_R = result_AA_R[quantile_th]
    postion_INTERPOLATED_5_5 = result_INTERPOLATED_5_5[quantile_th]
    postion_INTERPOLATED_7_3 = result_INTERPOLATED_7_3[quantile_th]
    postion_INTERPOLATED_3_7 = result_INTERPOLATED_3_7[quantile_th]
    postion_AND = result_AND[quantile_th]
    # 建立回傳dict
    result = {}
    result["method_single_f1"] = position_f1
    result["method_single_f2"] = position_f2
    result["method_AA"] = postion_AA
    result["method_AA_R"] = postion_AA_R
    # result["method_INTERPOLATED_5_5"] = postion_INTERPOLATED_5_5
    # result["method_INTERPOLATED_7_3"] = postion_INTERPOLATED_7_3
    # result["method_INTERPOLATED_3_7"] = postion_INTERPOLATED_3_7
    result["method_AND"] = postion_AND

    return result


# 取得五種不同雙因子方法的切割結果api(+內插加權法兩種不同全重)
def quantile_5_diff_method(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    quantile_th: int = 1,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
) -> Dict:
    """
    取得各自方法的Qunatile_Nth，
    Args:
        factor_name_list (list of str): 打算回測哪些因子，存入陣列中
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        data (get_data.data): 傳入價量資料，以便回測使用
        quantile (positive-int): 決定因子分析要切成幾個等份
        quantile_th (positive-int): 這次報告要產出第幾分位
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
    Returns:
        (Dict[str, pd.dataframe]): 記錄個雙因子模型的Quantile_n

    """
    # 執行因子分析
    result_AA = factor_analysis_two_factor_AA(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 不同權重的加權內插法
    f1 = factor_name_list[0]
    f2 = factor_name_list[1]
    # 第一種是 0.5:0.5
    result_INTERPOLATED_5_5 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )
    # 第二種全重 0.7:0.3
    factor_ratio_dict_1 = {}
    factor_ratio_dict_1[f1] = 0.7
    factor_ratio_dict_1[f2] = 0.3
    result_INTERPOLATED_7_3 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict_1,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )
    # 第三種全重 0.3:0.7
    factor_ratio_dict_2 = {}
    factor_ratio_dict_2[f1] = 0.3
    factor_ratio_dict_2[f2] = 0.7
    result_INTERPOLATED_3_7 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict_2,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )

    result_AND = factor_analysis_two_factor_AND(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 選取第幾分位
    quantile_th = "Quantile_" + str(quantile_th)
    postion_AA = result_AA[quantile_th]
    postion_INTERPOLATED_5_5 = result_INTERPOLATED_5_5[quantile_th]
    postion_INTERPOLATED_7_3 = result_INTERPOLATED_7_3[quantile_th]
    postion_INTERPOLATED_3_7 = result_INTERPOLATED_3_7[quantile_th]
    postion_AND = result_AND[quantile_th]
    # 建立回傳dict
    result = {}
    result["method_AA"] = postion_AA
    result["method_INTERPOLATED_5_5"] = postion_INTERPOLATED_5_5
    result["method_INTERPOLATED_7_3"] = postion_INTERPOLATED_7_3
    result["method_INTERPOLATED_3_7"] = postion_INTERPOLATED_3_7
    result["method_AND"] = postion_AND

    return result


# 取得五種不同雙因子方法的切割結果api(+內插加權所有的全重)
def quantile_13_diff_method(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    quantile: int = 4,
    quantile_th: int = 1,
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
) -> Dict:
    """
    取得各自方法的Qunatile_Nth，
    Args:
        factor_name_list (list of str): 打算回測哪些因子，存入陣列中
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        data (get_data.data): 傳入價量資料，以便回測使用
        quantile (positive-int): 決定因子分析要切成幾個等份
        quantile_th (positive-int): 這次報告要產出第幾分位
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
    Returns:
        (Dict[str, pd.dataframe]): 記錄個雙因子模型的Quantile_n

    """
    # 執行因子分析
    result_AA = factor_analysis_two_factor_AA(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 不同權重的加權內插法
    f1 = factor_name_list[0]
    f2 = factor_name_list[1]
    # 第一種是 0.5:0.5
    result_INTERPOLATED_5_5 = cal_factor_sum_df_interpolated(
        factor_name_list,
        factor_ratio_dict,
        factor_asc_dict,
        quantile,
        all_factor_df_dict,
    )

    result_AND = factor_analysis_two_factor_AND(
        factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    )
    # 選取第幾分位
    quantile_th = "Quantile_" + str(quantile_th)
    postion_AA = result_AA[quantile_th]
    postion_INTERPOLATED_5_5 = result_INTERPOLATED_5_5[quantile_th]
    postion_INTERPOLATED_7_3 = result_INTERPOLATED_7_3[quantile_th]
    postion_INTERPOLATED_3_7 = result_INTERPOLATED_3_7[quantile_th]
    postion_AND = result_AND[quantile_th]
    # 建立回傳dict
    result = {}
    result["method_AA"] = postion_AA
    result["method_INTERPOLATED_5_5"] = postion_INTERPOLATED_5_5
    result["method_INTERPOLATED_7_3"] = postion_INTERPOLATED_7_3
    result["method_INTERPOLATED_3_7"] = postion_INTERPOLATED_3_7
    result["method_AND"] = postion_AND

    return result


# 比較三種不同方法雙因子的績效api
def sim_3_diff_method(
    factor_name_list: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    data: Data,
    quantile: int = 4,
    quantile_th: int = 1,
    eval_indicator: str = "cum_returns",
    show: str = "multi",
    all_factor_df_dict: Dict[str, CustomDataFrame] = None,
) -> any:
    """
    比較三種不同方法雙因子的api
    Args:
        factor_name_list (list of str): 打算回測哪些因子，存入陣列中
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        data (get_data.data): 傳入價量資料，以便回測使用
        quantile (positive-int): 決定因子分析要切成幾個等份
        quantile_th (positive-int): 這次報告要產出第幾分位
        eval_indicator (str): 要劃出回測結果的哪個指標，例如:portfolio_value, portfolio_returns	, cum_returns, company_count
        show (str): 決定要...
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
    Returns:
        直接畫圖 或
        (tuples of pd.series): 記錄個雙因子模型的績效指標

    """
    # # 執行因子分析
    # result_AA = factor_analysis_two_factor_AA(
    #     factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    # )
    # result_INTERPOLATED = cal_factor_sum_df_interpolated(
    #     factor_name_list,
    #     factor_ratio_dict,
    #     factor_asc_dict,
    #     quantile,
    #     all_factor_df_dict,
    # )
    # result_AND = factor_analysis_two_factor_AND(
    #     factor_name_list, factor_asc_dict, quantile, all_factor_df_dict
    # )
    # # 選取第幾分位
    # quantile_th = "Quantile_" + str(quantile_th)
    # postion_AA = result_AA[quantile_th]
    # postion_INTERPOLATED = result_INTERPOLATED[quantile_th]
    # postion_AND = result_AND[quantile_th]

    result_dict = quantile_3_diff_method(
        factor_name_list,
        factor_ratio_dict,
        factor_asc_dict,
        quantile,
        quantile_th,
        all_factor_df_dict,
    )
    postion_AA = result_dict["method_AA"]
    postion_INTERPOLATED = result_dict["method_INTERPOLATED"]
    postion_AND = result_dict["method_AND"]
    # 執行回測
    sim_result_AA = sim(postion_AA, resample="Q", data=data)
    sim_result_INTERPOLATED = sim(postion_INTERPOLATED, resample="Q", data=data)
    sim_result_AND = sim(postion_AND, resample="Q", data=data)

    if show == "single":
        # 畫圖
        # 創建一個圖表和子圖
        fig, ax = plt.subplots()
        # 在第一個子圖中繪製第一個 DataFrame 的 "cum_returns"(eval_indicator) 列
        sim_result_AA.stock_data[eval_indicator].plot(ax=ax, label="aa_method_result")
        # 在第一個子圖中繪製第二個 DataFrame 的 "cum_returns"(eval_indicator) 列
        sim_result_INTERPOLATED.stock_data[eval_indicator].plot(
            ax=ax, label="interpolated_method_result"
        )
        # 在第一個子圖中繪製第四個 DataFrame 的 "cum_returns"(eval_indicator) 列
        sim_result_AND.stock_data[eval_indicator].plot(ax=ax, label="and_method_result")
        # 添加圖例
        ax.legend()
        # 顯示圖表
        plt.show()

    else:
        eval_indicator_AA = sim_result_AA.stock_data[eval_indicator]
        eval_indicator_AA.name = "method_AA"
        eval_indicator_INTERPOLATED = sim_result_INTERPOLATED.stock_data[eval_indicator]
        eval_indicator_INTERPOLATED.name = "method_INTERPOLATED"
        eval_indicator_AND = sim_result_AND.stock_data[eval_indicator]
        eval_indicator_AND.name = "method_AND"
        # 回傳tuple
        return (eval_indicator_AA, eval_indicator_INTERPOLATED, eval_indicator_AND)


# 畫出多個子圖比較雙因子
def two_factor_mulit_subplot(
    factor_name: List[str],
    factor_ratio_dict: Dict[str, float],
    factor_asc_dict: Dict[str, bool],
    all_factor_df_dict: Dict[str, CustomDataFrame],
    data: Data,
):
    """畫出雙因子之間的績效表現

    利用多個subplot呈現因子間相互關係

    Args:
        factor_name (list of str): 紀錄想要比較哪幾個因子的績效比現
        factor_ratio_dict (dict): 所有因子各自對印的比例(內積加權法用)
        factor_asc_dict (dict): 決定因子是越大(F)/小(T)越好, 因子的排序方向
        all_factor_df_dict (dict): dictionary中預先存放所有因子資料的DF集合，以便這輪需用到
        data (get_data.data): 傳入價量資料，以便回測使用

    Returns:
        None
    """
    # 因子的數量
    num_of_factor = len(factor_name)
    # 創建一個 NXN 的多圖布局
    fig, axes = plt.subplots(
        nrows=num_of_factor,
        ncols=num_of_factor,
        figsize=(20, 15),
        sharex=True,
        sharey=True,
    )

    # 使用迴圈遍歷每個子圖
    for i in range(num_of_factor):
        for j in range(num_of_factor):
            # if i == j:
            #     continue
            # 記錄第1、2個因子
            factor_1 = factor_name[i]
            factor_2 = factor_name[j]
            factor_name_list = []
            factor_name_list = [factor_1, factor_2]
            print("目前正在實作:", factor_1, "+", factor_2, "因子組合")
            # factor_df_dict = {}
            # factor_df_dict = generate_factor_df(factor_1, factor_2)
            # print("factor_df_dict", factor_df_dict)
            # 呼叫回測三種雙因子的API
            (
                eval_indicator_AA,
                eval_indicator_INTERPOLATED,
                eval_indicator_AND,
            ) = sim_3_diff_method(
                factor_name_list,
                factor_ratio_dict,
                factor_asc_dict,
                data,
                quantile=4,
                quantile_th=1,
                # eval_indicator="cum_returns",
                eval_indicator="company_count",
                show="multi",
                all_factor_df_dict=all_factor_df_dict,
            )
            # 在每個子圖上畫折線圖，將 data 改成 custom_data
            sns.lineplot(
                x=eval_indicator_AA.index,
                y=eval_indicator_AA,
                ax=axes[i, j],
                label=eval_indicator_AA.name,
            )
            sns.lineplot(
                x=eval_indicator_INTERPOLATED.index,
                y=eval_indicator_INTERPOLATED,
                ax=axes[i, j],
                label=eval_indicator_INTERPOLATED.name,
            )
            sns.lineplot(
                x=eval_indicator_AND.index,
                y=eval_indicator_AND,
                ax=axes[i, j],
                label=eval_indicator_AND.name,
            )

            # 可以自行調整子圖的標題等
            axes[i, j].set_title("Factor_" + factor_name[i] + "_" + factor_name[j])

    # 調整整體布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
    # # 建立取得資料物件
    # # data = Data()
    # factor_name = ["EPS", "PE"]

    # # 自動建立因子DF
    # all_factor_df_dict = {}
    # for f in factor_name:
    #     key = f
    #     value = pd.read_csv("./OutputFile/factor_data/" + key + ".csv", index_col=0)
    #     all_factor_df_dict[key] = CustomDataFrame(value)

    # # 建立因子權重、排序方向
    # factor_ratio_dict = {
    #     "EPS": 0.5,
    #     "PE": 0.5,
    #     "ROE": 0.5,
    #     "PB": 0.5,
    #     "PS": 0.5,
    #     "ROIC": 0.5,
    # }
    # factor_asc_dict = {
    #     "EPS": False,
    #     "PE": True,
    #     "ROE": False,
    #     "PB": True,
    #     "PS": True,
    #     "ROIC": False,
    # }
    # result = factor_analysis_multi_dual(
    #     factor_name, factor_ratio_dict, factor_asc_dict, 4, all_factor_df_dict, "AA"
    # )

    # # 繪製各QUANTILE入選數量比較圖
    # plot_multi_lines_from_dict(result)
