def get(dataset: str, save_to_storage: bool = True, force_download=False):
    """下載歷史資料

    請至[歷史資料目錄](https://ai.finlab.tw/database) 來獲得所有歷史資料的名稱，即可使用此函式來獲取歷史資料。
    假設 `save_to_storage` 為 `True` 則，程式會自動在本地複製一份，以避免重複下載大量數據。

    Args:
        dataset (str): The name of dataset.
        save_to_storage (bool): Whether to save the dataset to storage for later use.

    Returns:
        (pd.DataFrame): financial data

    Examples:

        欲下載所有上市上櫃之收盤價歷史資料，只需要使用此函式即可:

        ``` py
        from finlab import data
        close = data.get('price:收盤價')
        close
        ```

        | date       |   0015 |   0050 |   0051 |   0052 |   0053 |
        |:-----------|-------:|-------:|-------:|-------:|-------:|
        | 2007-04-23 |   9.54 |  57.85 |  32.83 |  38.4  |    nan |
        | 2007-04-24 |   9.54 |  58.1  |  32.99 |  38.65 |    nan |
        | 2007-04-25 |   9.52 |  57.6  |  32.8  |  38.59 |    nan |
        | 2007-04-26 |   9.59 |  57.7  |  32.8  |  38.6  |    nan |
        | 2007-04-27 |   9.55 |  57.5  |  32.72 |  38.4  |    nan |

    """
    check_version()

    global universe_stocks
    global _storage

    # 這些好像是免費板可以用的(不包含VIP資料)
    not_available_universe_stocks = [
        'benchmark_return', 'institutional_investors_trading_all_market_summary',
        'margin_balance', 'intraday_trading_stat',
        'stock_index_price', 'stock_index_vol',
        'taiex_total_index', 'broker_info',
        'rotc_monthly_revenue', 'rotc_price',
        'world_index', 'rotc_broker_trade_record',
        'security_categories', 'finlab_tw_stock_market_ind',
        'tw_industry_pmi', 'tw_industry_nmi',
        'tw_total_pmi', 'tw_total_nmi',
        'tw_business_indicators', 'tw_business_indicators_details',
        'tw_monetary_aggregates', 'us_unemployment_rate_seasonally_adjusted',
        'us_tickers',
        ]

    def refine_stock_id(ret):

        if dataset.split(':')[0] in not_available_universe_stocks:
            return ret
        if ':' in dataset:
            return ret if not universe_stocks else ret[ret.columns.intersection(universe_stocks)]
        if 'stock_id' in ret.columns:
            return ret if not universe_stocks else ret.loc[ret['stock_id'].isin(universe_stocks)]

        return ret

    # not expired
    time_expired = _storage.get_time_expired(dataset)
    if time_expired and time_expired > CacheStorage.now() and not force_download:
        return refine_stock_id(finlab.dataframe.FinlabDataFrame(_storage.get_dataframe(dataset)))

    # request for auth url
    url = 'https://asia-east2-fdata-299302.cloudfunctions.net/auth_generate_data_url'
    params = {
        'api_token': finlab.get_token(),
        'bucket_name': 'finlab_tw_stock_item',
        'blob_name': dataset.replace(':', '#') \
                + ('.pickle' if "pyodide" in sys.modules else '.feather'),
        'pyodide': 'pyodide' in sys.modules
    }

    time_saved = _storage.get_time_created(dataset)
    if time_saved and not force_download:
        params['time_saved'] = time_saved.strftime('%Y%m%d%H%M%S')

    res = requests.post(url, params)

    try:
        url_data = res.json()
    except:
        raise Exception("Cannot get response from data server.")

    # use cache
    global has_print_free_user_warning
    if not has_print_free_user_warning \
            and 'role' in url_data \
            and url_data['role'] == 'free':
        logger.warning('Due to your status as a free user,'
            'the most recent data has been shortened or limited.')
        has_print_free_user_warning = True

    if 'url' in url_data and url_data['url'] == '':
        return refine_stock_id(finlab.dataframe.FinlabDataFrame(_storage.get_dataframe(dataset)))

    if 'quota' in url_data:
        logger.warning(
            f'{dataset} -- Daily usage: {url_data["quota"]:.1f} / {url_data["limit_size"]} MB')



    if 'error' in url_data:

        if url_data['error'] in [
            'request not valid',
            'User not found',
            'api_token not valid',
                'api_token not match', ]:
            finlab.login()
            return get(dataset, save_to_storage)

        raise Exception(f"**Error: {url_data['error']}")

    assert 'url' in url_data

    if 'pyodide' in sys.modules:
        if hasattr(requests, 'getBytes'):
            res = requests.getBytes(url_data['url'])
            df = pd.read_pickle(BytesIO(res), compression='gzip')
        else:
            res = requests.get(url_data['url'])
            df = pd.read_pickle(BytesIO(res.content), compression='gzip')
    else:
        res = requests.get(url_data['url'])
        df = pd.read_feather(BytesIO(res.content))

    # set market type on column name

    if ':' in dataset:
        df.columns.name = f'symbol'

    # set date as index
    if 'date' in df:
        df.set_index('date', inplace=True)

        table_name = dataset.split(':')[0]
        if table_name in ['tw_total_pmi', 'tw_total_nmi', 'tw_industry_nmi', 'tw_industry_pmi']:
            if isinstance(df.index[0], pd.Timestamp):
                close = get('price:收盤價')
                df.index = df.index.map(
                    lambda d: d if len(close.loc[d:]) == 0 or d < close.index[0] else close.loc[d:].index[0])

        # if column is stock name
        if (df.columns.str.find(' ') != -1).all():

            # remove stock names
            df.columns = df.columns.str.split(' ').str[0]

            # combine same stock history according to sid
            check_numeric_dtype = pd.api.types.is_numeric_dtype(df.values)
            if check_numeric_dtype:
                df = df.transpose().groupby(level=0).mean().transpose()
            else:
                df = df.fillna(np.nan).transpose().groupby(
                    level=0).last().transpose()

        df = finlab.dataframe.FinlabDataFrame(df)

        if table_name in ['monthly_revenue', 'rotc_monthly_revenue']:
            df = df._index_to_business_day()
        elif table_name in ['financial_statement', 'fundamental_features',]:
            df = df._index_date_to_str_season()
        elif table_name in ['us_fundamental', 'us_fundamental_ART']:
            df = df._index_date_to_str_season('-US')
        elif table_name in ['us_fundamental_all', 'us_fundamental_all_ART']:
            df = df._index_date_to_str_season('-US-ALL')
    # save cache
    if save_to_storage:
        expiry = datetime.datetime.strptime(
                url_data['time_scheduled'], '%Y%m%d%H%M%S').replace(tzinfo=datetime.timezone.utc)\
                if 'time_scheduled' in url_data else None

        _storage.set_dataframe(dataset, df, expiry=expiry)

    return refine_stock_id(df)