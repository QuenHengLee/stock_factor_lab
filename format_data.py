# format_data.py
# 從原本get_data.py節取出來的，更模組化
# 這些function是將原始DB TABLE的資料轉成get() api可用格式

import pandas as pd

def format_price_data(raw_price_data, item):
    selected_data = raw_price_data[["date", item, "company_symbol"]]
    pivot_data = selected_data.pivot_table(
        index="date", columns="company_symbol", values=item
    )
    return pivot_data

def format_report_data(raw_report_data, factor):
    unique_ids = raw_report_data["factor_name"].unique()
    dfs_by_id = {}
    for unique_id in unique_ids:
        temp_df = raw_report_data[
            raw_report_data["factor_name"] == unique_id
        ].pivot(index="date", columns="company_symbol", values="factor_value")
        dfs_by_id[unique_id] = temp_df

    return dfs_by_id[factor]

def handle_price_data(raw_price_data):
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

    # 使用布林索引過濾 DataFrame，擷取 "加工業" 種類的收盤價資料
    filtered_df = raw_price_data[raw_price_data["company_symbol"] == company_symbol]
    filtered_df.set_index("date", inplace=True)
    # 升序排序日期索引
    filtered_df = filtered_df.sort_index(ascending=True)

    return filtered_df

def get_all_company_symbol(raw_price_data):
    # 找出 "symbol" 列中的所有不重複值
    unique_symbols = raw_price_data["company_symbol"].drop_duplicates()
    unique_symbols_list = unique_symbols.tolist()

    return unique_symbols_list
