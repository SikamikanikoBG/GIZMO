import numpy as np
from stockstats import StockDataFrame as Sdf


def run(df):
    df = df.drop_duplicates(subset=['time'], keep='last')
    df = df.tail(40000)
    df = df.reset_index(drop=True)
    column_names_to_forex_rename(df)
    calculate_indicators(df)

    return df


def calculate_criterion(df):
    df['check_ROLL_min_90'] = df['close'][::-1].rolling(window=90).min()
    df['check_ROLL_max_90'] = df['close'][::-1].rolling(window=90).max()

    df['criterion_ROLLING_min_90'] = df['close'][::-1].rolling(window=90).min() - df['open']
    df['criterion_ROLLING_max_90'] = df['close'][::-1].rolling(window=90).max() - df['open']

    df['criterion_buy'] = np.where((df['criterion_ROLLING_max_90'] > .00030)
                                   & (df['criterion_ROLLING_min_90'] > -.00050), 1, 0)
    df['criterion_sell'] = np.where((df['criterion_ROLLING_max_90'] < .00050)
                                    & (df['criterion_ROLLING_min_90'] < -.00030), 1, 0)
    return df


def column_names_to_forex_rename(df):
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'real_volume', 'Currency']
    return df


def columns_to_lower_case_names(df):
    df.columns = df.columns.str.lower()
    return df


def calculate_indicators(df):
    stock_df = Sdf.retype(df)
    #periods = ["14", "60", "240", "480"]
    periods = ["14", "240"]
    columns = ['open', 'high', 'low', 'close', 'volume']
    #columns = ['open', 'high', 'low', 'close', 'volume', 'spread']
    for period in periods:
        for col in columns:
            df[col + '_' + period + '_ema'] = stock_df[col + '_' + period + '_ema']
            df[col + '_' + period + '_ema'] = stock_df[col + '_' + period + '_ema']
            df[col + '_' + period + '_delta'] = stock_df[col + '_-' + period + '_d']
            df[col + '_' + period + '_smma'] = stock_df[col + '_' + period + '_smma']
            #df[col + '_' + period + '_ups_c'] = stock_df['ups_' + period + '_c']
            #df[col + '_' + period + '_downs_c'] = stock_df['downs_' + period + '_c']
        df['rsi_'+period] = stock_df['rsi_'+period]
        df['vr_' + period] = stock_df['vr_' + period]
        df['wr_' + period] = stock_df['wr_' + period]
        df['wr_' + period] = stock_df['wr_' + period]
        df['cci_' + period] = stock_df['cci_' + period]
        df['atr_' + period] = stock_df['atr_' + period]
        df['middle_'+period+'_trix'] = stock_df['middle_'+period+'_trix']
        df['middle_' + period + '_tema'] = stock_df['middle_' + period + '_tema']
        df['kdjk_' + period] = stock_df['kdjk_' + period]
        df['vwma_' + period] = stock_df['vwma_' + period]

    df['dma'] = stock_df['dma']
    df['pdi'] = stock_df['pdi']
    df['mdi'] = stock_df['mdi']
    df['dx'] = stock_df['dx']
    df['adx'] = stock_df['adx']
    df['adxr'] = stock_df['adxr']

    df['cr'] = stock_df['cr']
    df['ppo'] = stock_df['ppo']
    df['log-ret'] = stock_df['log-ret']
    df['macd_feat'] = stock_df['macd']
    df['macds_feat'] = stock_df['macds']
    df['boll_feat'] = stock_df['boll']
    df['boll_ub_feat'] = stock_df['boll_ub']
    df['boll_lb_feat'] = stock_df['boll_lb']
    del stock_df
    return df
