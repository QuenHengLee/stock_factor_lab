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
