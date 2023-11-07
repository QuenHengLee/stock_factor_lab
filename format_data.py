# 從原本get_data.py節取出來的，更模組化
# 這些function是將原始DB TABLE的資料轉成get() api可用格式

import pandas as pd

def format_price_data(raw_price_data, item):
    """
    格式化原始價格資料成樞紐表格式。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。
    item (str)：要提取的項目（例如："open"、"high"、"low"、"close"、"volume"、"market_capital"）。

    返回：
    DataFrame：包含每家公司和日期的項目值的樞紐表。
    """
    selected_data = raw_price_data[["date", item, "company_symbol"]]
    pivot_data = selected_data.pivot_table(
        index="date", columns="company_symbol", values=item
    )
    return pivot_data

def format_report_data(raw_report_data, factor):
    """
    格式化原始報告資料成不同因子的DataFrame字典。

    參數：
    raw_report_data (DataFrame)：來自DB表格的原始報告資料。
    factor (str)：要提取的因子名稱。

    返回：
    dict：包含每家公司和日期的因子值的DataFrame字典。
    """
    unique_ids = raw_report_data["factor_name"].unique()
    dfs_by_id = {}
    for unique_id in unique_ids:
        temp_df = raw_report_data[
            raw_report_data["factor_name"] == unique_id
        ].pivot(index="date", columns="company_symbol", values="factor_value")
        dfs_by_id[unique_id] = temp_df

    return dfs_by_id[factor]

def handle_price_data(raw_price_data):
    """
    格式化原始價格資料成包含不同價格項目的字典。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。

    返回：
    dict：包含"open"、"high"、"low"、"close"、"volume"和"market_capital"的DataFrame的字典。
    """
    all_open = format_price_data(raw_price_data, "open")
    all_high = format_price_data(raw_price_data, "high")
    all_low = format_price_data(raw_price_data, "low")
    all_close = format_price_data(raw_price_data, "close")
    all_volume = format_price_data(raw_price_data, "volume")
    all_market_capital = format_price_data(raw_price_data, "market_capital")
    all_price_dict = {
        "open": all_open,
        "high": all_high,
        "low": all_low,
        "close": all_close,
        "volume": all_volume,
        "market_capital": all_market_capital,
    }
    return all_price_dict

def get_each_company_daily_price(raw_price_data, company_symbol):
    """
    獲取特定公司的每日價格資料。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。
    company_symbol (str)：要檢索資料的公司代號。

    返回：
    DataFrame：指定公司的每日價格資料。
    """
    filtered_df = raw_price_data[raw_price_data["company_symbol"] == company_symbol]
    filtered_df.set_index("date", inplace=True)
    filtered_df = filtered_df.sort_index(ascending=True)
    return filtered_df

def get_all_company_symbol(raw_price_data):
    """
    獲取原始價格資料中所有獨特的公司代號清單。

    參數：
    raw_price_data (DataFrame)：來自DB表格的原始價格資料。

    返回：
    list：獨特的公司代號清單。
    """
    unique_symbols = raw_price_data["company_symbol"].drop_duplicates()
    unique_symbols_list = unique_symbols.tolist()
    return unique_symbols_list
