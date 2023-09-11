def get_indicator_value(self, indname, adjust_price=False, resample='D', market='TW_STOCK', **kwargs):
    # 宣告一個DF站存指標數據(複製是為了INDEX日期一致)
    tmp_dataframe = self.all_data_dict['close'].copy()
    print('指標:'+indname+' 參數:',kwargs)
    return_data =[]
    # 瀏覽所有公司
    for column in self.all_data_dict['close'].columns:
        # 取得存放單一公司開高低收的DF
        inputs = self.get_daily_stock_list(column)
        if column != 'Date':
            # 實際呼叫TALIB計算指標
            output=  abstract.Function('bbands')(inputs, kwargs.get('timeperiod',20))
            # 計算回傳多少個DF (EX: SMA 一個; BBANDS 三個)
            num_columns = output.shape[1]
            # 用list存放有哪些結果 (EX: BBANDS的UPPER,MIDDLE,LOWER)
            column_names = output.columns.tolist()
            for cn in column_names:
                # output = pd.DataFrame(output)
                tmp_output = pd.DataFrame(output)[cn]
                print(cn)
                print(tmp_output)
                # 前面用Dataframe比較好用index取值，但要存入則需轉回list
                tmp_output = tmp_output.tolist()
                tmp_dataframe[f'{column}'] = tmp_output
                return_data.append(tmp_dataframe)
    return return_data