import numpy as np
from stockstats import StockDataFrame as Sdf
import definitions
from src.functions.printing_and_logging import print_and_log


def run(df):
    # df = df.drop_duplicates(subset=['time'], keep='last')
    # df = df.tail(40000)
    # df = df.reset_index(drop=True)

    column_names_to_forex_rename(df)
    df.sort_values(by=['Time'], inplace=True)
    calculate_indicators(df)
    #df['time'] = df['time'].astype("O")
    return df


def calculate_criterion(df, predict_module):
    profit_pips = float(definitions.args.tp)
    loss_pips = float(definitions.args.sl) #.0050
    period = int(definitions.args.period) #480

    df['check_ROLL_min'] = df['low'][::-1].rolling(window=period).min()
    df['check_ROLL_max'] = df['high'][::-1].rolling(window=period).max()
    df['check_pips'] = df['open'] + profit_pips

    df['criterion_ROLLING_min'] = df['low'][::-1].rolling(window=period).min() - df['open']
    df['criterion_ROLLING_max'] = df['high'][::-1].rolling(window=period).max() - df['open']

    df['criterion_buy'] = np.where((df['criterion_ROLLING_max'] > profit_pips)
                                   & (df['criterion_ROLLING_min'] > -loss_pips), 1, 0)
    df['criterion_sell'] = np.where((df['criterion_ROLLING_max'] < loss_pips)
                                    & (df['criterion_ROLLING_min'] < -profit_pips), 1, 0)

    criterion_buy = df['criterion_buy'].sum() / df['criterion_buy'].count()
    criterion_sell = df['criterion_sell'].sum() / df['criterion_sell'].count()

    print_and_log(f"Criterion BUY: {criterion_buy}, "
          f"Criterion SELL: {criterion_sell}", "")

    if predict_module:
        pass
    elif df['criterion_buy'].sum() == 0 or df['criterion_sell'].sum()==0 or df['criterion_buy'].sum() == 1 or df['criterion_sell'].sum()==1:
        print_and_log(f"ERROR: Only one value in Criterion. Quitting!", "RED")
        quit()
    else:
        pass
    return df


def column_names_to_forex_rename(df):
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'real_volume', 'Currency']
    return df


def columns_to_lower_case_names(df):
    df.columns = df.columns.str.lower()
    return df


def calculate_indicators(df):
    stock_df = Sdf.retype(df)
    # periods = ["14", "60", "240", "480"]
    periods = ["14", "240", "480"]
    columns = ['open']
    # columns = ['open', 'high', 'low', 'close', 'volume', 'spread']
    for period in periods:
        for col in columns:
            df[col + '_' + period + '_ema'] = stock_df[col + '_' + period + '_ema']
            #df[col + '_' + period + '_delta'] = stock_df[col + '_-' + period + '_d']
            #df[col + '_' + period + '_smma'] = stock_df[col + '_' + period + '_smma']
            #df[col + '_' + period + '_ups_c'] = stock_df['ups_' + period + '_c']
            #df[col + '_' + period + '_downs_c'] = stock_df['downs_' + period + '_c']
        df['rsi_' + period] = stock_df['rsi_' + period]
        df['vr_' + period] = stock_df['vr_' + period]
        df['wr_' + period] = stock_df['wr_' + period]
        df['cci_' + period] = stock_df['cci_' + period]
        df['atr_' + period] = stock_df['atr_' + period]
        #df['middle_' + period + '_trix'] = stock_df['middle_' + period + '_trix']
        #df['middle_' + period + '_tema'] = stock_df['middle_' + period + '_tema']
        #df['kdjk_' + period] = stock_df['kdjk_' + period]
        #df['vwma_' + period] = stock_df['vwma_' + period]

    df['dma'] = stock_df['dma']
    df['pdi'] = stock_df['pdi']
    df['mdi'] = stock_df['mdi']
    df['dx'] = stock_df['dx']
    df['adx'] = stock_df['adx']
    df['adxr'] = stock_df['adxr']

    #df['cr'] = stock_df['cr']
    #df['ppo'] = stock_df['ppo']
    df['log-ret'] = stock_df['log-ret']
    df['macd_feat'] = stock_df['macd']
    df['macds_feat'] = stock_df['macds']
    df['boll_feat'] = stock_df['boll']
    df['boll_ub_feat'] = stock_df['boll_ub']
    df['boll_lb_feat'] = stock_df['boll_lb']
    del stock_df
    return df
