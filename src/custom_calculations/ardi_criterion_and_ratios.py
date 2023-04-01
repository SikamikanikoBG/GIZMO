import numpy as np
from stockstats import StockDataFrame as Sdf
import definitions
from src.functions.printing_and_logging import print_and_log
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def run(df):
    # df = df.drop_duplicates(subset=['time'], keep='last')
    # df = df.tail(40000)
    # df = df.reset_index(drop=True)

    column_names_to_forex_rename(df)
    df.sort_values(by=['Time'], inplace=True)
    calculate_indicators(df)
    # df['time'] = df['time'].astype("O")
    return df


def calculate_flag_trend(df):
    # Load ma simulation file to extract parameters from it
    ma_best_simu_results = pd.read_csv(f"{definitions.EXTERNAL_DIR}/ma_simu/ma_best_simu_results.csv")
    project = definitions.args.project.lower()
    _, currency, direction = project.split("_")

    ma_best_simu_results = ma_best_simu_results[ma_best_simu_results['currency'] == currency.upper()].copy()
    ma_best_simu_results = ma_best_simu_results[ma_best_simu_results['direction'] == direction].copy()

    timeframe_fast = ma_best_simu_results['timeframe_fast'].iloc[-1]
    timeframe_slow = ma_best_simu_results['timeframe_slow'].iloc[-1]
    period_fast_small = ma_best_simu_results['period_fast_small'].iloc[-1]
    period_fast_big = ma_best_simu_results['period_fast_big'].iloc[-1]
    period_slow = ma_best_simu_results['period_slow'].iloc[-1]
    diff_pips = ma_best_simu_results['diff_pips'].iloc[-1]

    # Aggregate 1M to 15M df
    df7 = df.iloc[::-timeframe_slow, :].copy()
    df7 = df7.iloc[::-1].copy()
    # df15.sort_values(by=['Time'], inplace=True)
    stock_df7 = Sdf.retype(df7)
    stock_df = Sdf.retype(df)

    # Calculate flag for trends
    # open_2M_smma = stock_df['open_2_sma'].iloc[-1]
    open_21M_smma = stock_df[f'open_{period_fast_small}_sma'].iloc[-1]
    open_27M_smma = stock_df7[f'open_{period_fast_big}_sma'].iloc[-1]
    open_47M_smma = stock_df7[f'open_{period_slow}_sma'].iloc[-1]
    abs_diff_pips = abs(open_21M_smma - open_47M_smma)
    abs_diff_pips_fast = open_21M_smma - open_27M_smma

    # trends
    flag_trend = 0
    if "buy" in definitions.args.project.lower():
        if (open_21M_smma > open_47M_smma) and (open_27M_smma > open_47M_smma) and (abs_diff_pips > diff_pips) and (
                abs_diff_pips_fast > -diff_pips):
            flag_trend = 1
    elif "sell" in definitions.args.project.lower():
        if (open_21M_smma < open_47M_smma) and (open_27M_smma < open_47M_smma) and (abs_diff_pips > diff_pips) and (
                abs_diff_pips_fast < diff_pips):
            flag_trend = 1

    df['flag_trend'] = flag_trend
    del stock_df7

    # todo: check for errors, remove after not used

    test = pd.DataFrame(
        columns=['time', 'proj', 'mafast', 'mafastbig', 'maslowbig', 'diff', 'diff_fast', 'flag', 'timeframe_slow',
                 'period_fast_small', 'period_slow'])
    test = test.append({'time': datetime.now(),
                        'proj': definitions.args.project,
                        'mafast': open_21M_smma,
                        'mafastbig': open_27M_smma,
                        'maslowbig': open_47M_smma,
                        'diff': abs_diff_pips,
                        'diff_fast': abs_diff_pips_fast,
                        'flag': flag_trend,
                        'timeframe_slow': timeframe_slow,
                        'period_fast_small': period_fast_small,
                        'period_fast_big': period_fast_big,
                        'period_slow': period_slow
                        }, ignore_index=True)
    test.to_csv(f"{definitions.EXTERNAL_DIR}/ma_simu/test_{definitions.args.project}.csv", index=False)

    return df


def calculate_criterion(df, predict_module):
    profit_pips = float(definitions.args.tp)
    loss_pips = float(definitions.args.sl)  # .0050
    period = int(definitions.args.period)  # 480

    if "jpy" in definitions.args.project.lower():
        profit_pips = profit_pips * 100
        loss_pips = loss_pips * 100
    elif "us30" in definitions.args.project.lower():
        profit_pips = profit_pips * 1000
        loss_pips = loss_pips * 1000
    elif "xau" in definitions.args.project.lower():
        profit_pips = profit_pips * 1000
        loss_pips = loss_pips * 1000
    elif "xag" in definitions.args.project.lower():
        profit_pips = profit_pips * 100
        loss_pips = loss_pips * 100
    else:
        pass

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

    if definitions.params:
        if definitions.params["resistance_support"]:
            r1 = float(definitions.params["resistance_support"]["r1"])
            r2 = float(definitions.params["resistance_support"]["r2"])
            r3 = float(definitions.params["resistance_support"]["r3"])
            s1 = float(definitions.params["resistance_support"]["s1"])
            s2 = float(definitions.params["resistance_support"]["s2"])
            s3 = float(definitions.params["resistance_support"]["s3"])

            df["open_to_r1"] = df["open"] / r1
            df["open_to_r2"] = df["open"] / r2
            df["open_to_r3"] = df["open"] / r3
            df["open_to_s1"] = df["open"] / s1
            df["open_to_s2"] = df["open"] / s2
            df["open_to_s3"] = df["open"] / s3

    if predict_module:
        pass
    elif df['criterion_buy'].sum() == 0 or df['criterion_sell'].sum() == 0 or df['criterion_buy'].sum() == 1 or df[
        'criterion_sell'].sum() == 1:
        print_and_log(f"ERROR: Only one value in Criterion. Quitting!", "RED")
        quit()
    else:
        pass

    df = calculate_flag_trend(df)

    return df


def column_names_to_forex_rename(df):
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 'real_volume', 'Currency']
    return df


def columns_to_lower_case_names(df):
    df.columns = df.columns.str.lower()
    return df


def calculate_indicators(df):
    # Retype df
    stock_df = Sdf.retype(df)

    # periods = ["14", "60", "240", "480"]
    periods = ["14", "240", "480"]
    columns = ['open']
    # columns = ['open', 'high', 'low', 'close', 'volume', 'spread']
    for period in periods:
        for col in columns:
            df[col + '_' + period + '_ema'] = stock_df[col + '_' + period + '_ema']
            # df[col + '_' + period + '_delta'] = stock_df[col + '_-' + period + '_d']
            # df[col + '_' + period + '_smma'] = stock_df[col + '_' + period + '_smma']
            # df[col + '_' + period + '_ups_c'] = stock_df['ups_' + period + '_c']
            # df[col + '_' + period + '_downs_c'] = stock_df['downs_' + period + '_c']
        df['rsi_' + period] = stock_df['rsi_' + period]
        df['vr_' + period] = stock_df['vr_' + period]
        df['wr_' + period] = stock_df['wr_' + period]
        # df['cci_' + period] = stock_df['cci_' + period]
        df['atr_' + period] = stock_df['atr_' + period]
        # df['middle_' + period + '_trix'] = stock_df['middle_' + period + '_trix']
        # df['middle_' + period + '_tema'] = stock_df['middle_' + period + '_tema']
        # df['kdjk_' + period] = stock_df['kdjk_' + period]
        # df['vwma_' + period] = stock_df['vwma_' + period]

    df['dma'] = stock_df['dma']
    df['pdi'] = stock_df['pdi']
    df['mdi'] = stock_df['mdi']
    df['dx'] = stock_df['dx']
    df['adx'] = stock_df['adx']
    df['adxr'] = stock_df['adxr']

    # df['cr'] = stock_df['cr']
    # df['ppo'] = stock_df['ppo']
    df['log-ret'] = stock_df['log-ret']
    df['macd_feat'] = stock_df['macd']
    df['macds_feat'] = stock_df['macds']
    df['boll_feat'] = stock_df['boll']
    df['boll_ub_feat'] = stock_df['boll_ub']
    df['boll_lb_feat'] = stock_df['boll_lb']

    del stock_df

    return df


# Calculations for MA simulation

def calculate_flag_trend_ma_simu(df, timeframe_fast, timeframe_slow, period_fast_small, period_fast_big, period_slow, diff_pips, direction):
    # Aggregate 1M to 15M df
    df['direction'] = direction
    df_org = df.copy()
    df = df_org.iloc[::-timeframe_fast, :].copy()
    df = df.iloc[::-1].copy()

    df_slow = df_org.iloc[::-timeframe_slow, :].copy()
    df_slow = df_slow.iloc[::-1].copy()
    # df15.sort_values(by=['Time'], inplace=True)
    stock_df_slow = Sdf.retype(df_slow)
    stock_df = Sdf.retype(df)

    # Calculate flag for trends
    stock_df['open_21m_smma'] = stock_df[f'open_{period_fast_small}_sma']
    stock_df_slow['open_27m_smma'] = stock_df_slow[f'open_{period_fast_big}_sma']
    stock_df_slow['open_47m_smma'] = stock_df_slow[f'open_{period_slow}_sma']

    stock_df = stock_df.merge(stock_df_slow[['time', 'open_27m_smma', 'open_47m_smma']], on='time', how='left')
    stock_df.fillna(method='ffill', inplace=True)

    # display(stock_df.head(5))

    stock_df['abs_diff_pips'] = stock_df['open_21m_smma'] - stock_df['open_47m_smma']
    stock_df['abs_diff_pips'] = abs(stock_df['abs_diff_pips'])
    stock_df['abs_diff_pips_fast'] = stock_df['open_21m_smma'] - stock_df['open_27m_smma']

    # trends
    stock_df['flag_trend'] = 0

    if "buy" in direction.lower():
        stock_df['flag_trend'] = np.where(
            ((stock_df['open_21m_smma'] > stock_df['open_47m_smma'])
             & (stock_df['open_27m_smma'] > stock_df['open_47m_smma'])
             & (stock_df['abs_diff_pips'] > diff_pips)
             & (stock_df['abs_diff_pips_fast'] > -diff_pips)), 1, 0)
        # if (open_21M_smma > open_47M_smma) and (open_27M_smma > open_47M_smma) and (abs_diff_pips > diff_pips):
        #    flag_trend = 1
    elif "sell" in direction.lower():
        stock_df['flag_trend'] = np.where(
            ((stock_df['open_21m_smma'] < stock_df['open_47m_smma'])
             & (stock_df['open_27m_smma'] < stock_df['open_47m_smma'])
             & (stock_df['abs_diff_pips'] > diff_pips)
             & (stock_df['abs_diff_pips_fast'] < diff_pips)), 1, 0)
        # if (open_21M_smma < open_47M_smma) and (open_27M_smma < open_47M_smma) and (abs_diff_pips > diff_pips):
        #    flag_trend = 1

    del stock_df_slow
    return stock_df


def calculate_criterion_ma_simu(df, predict_module, currency, tp, sl, period):
    profit_pips = float(tp)
    loss_pips = float(sl)  # .0050
    period = int(period)  # 480

    if "jpy" in currency:
        profit_pips = profit_pips * 100
        loss_pips = loss_pips * 100
    elif "us30" in currency:
        profit_pips = profit_pips * 1000
        loss_pips = loss_pips * 1000
    elif "xau" in currency:
        profit_pips = profit_pips * 1000
        loss_pips = loss_pips * 1000
    elif "xag" in currency:
        profit_pips = profit_pips * 100
        loss_pips = loss_pips * 100
    else:
        pass

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

    # print(f"Criterion BUY: {criterion_buy}, "
    #              f"Criterion SELL: {criterion_sell}", "")

    # calculate profit
    df['profit_buy'] = 0
    df['profit_buy'] = np.where((df['criterion_buy'] == 1)
                                & (df['flag_trend'] == 1), tp, df['profit_buy'])
    df['profit_buy'] = np.where((df['criterion_buy'] == 0)
                                & (df['flag_trend'] == 1), -sl, df['profit_buy'])

    df['profit_sell'] = 0
    df['profit_sell'] = np.where((df['criterion_sell'] == 1)
                                 & (df['flag_trend'] == 1), tp, df['profit_sell'])
    df['profit_sell'] = np.where((df['criterion_sell'] == 0)
                                 & (df['flag_trend'] == 1), -sl, df['profit_sell'])

    return df


def ma_simulation(source_df):
    timeframe_fast = [1]
    timeframe_slow = [1, 2, 15]
    period_fast_small = [5, 10, 20, 30]
    period_fast_big = period_fast_small.copy() # The idea is to use the same values on different timeframes separately
    period_slow = [5, 15, 30, 60]
    diff_pips = [0.0005, 0.0010]
    direction = ['buy', 'sell']
    currency_list = source_df['Currency'].unique().tolist()
    for el in currency_list[:]:
        if ('jpy' in el.lower()) or ('xau' in el.lower()) or ('xag' in el.lower()) or ('us30' in el.lower()):
            currency_list.remove(el)

    sl = [0.005, 0.003]

    ma_best_simu_results = pd.DataFrame()

    for currency_list_el in currency_list:
        result = pd.DataFrame(
            columns=['currency', 'time_min', 'time_max', 'timeframe_fast', 'timeframe_slow', 'period_fast_small', 'period_fast_big',
                     'period_slow', 'diff_pips','direction', 'sl', 'profit_buy', 'profit_sell'])
        for timeframe_fast_el in timeframe_fast:
            for timeframe_slow_el in timeframe_slow:
                for period_fast_el in period_fast_small:
                    for period_fast_big_el in period_fast_big:
                        for period_slow_el in period_slow:
                            if period_fast_big_el < period_slow_el: # in order on bigger timeframes the faster to be faster than the slower :)
                                for diff_pips_el in diff_pips:
                                    for direction_el in direction:
                                        for sl_el in sl:
                                            test = source_df[source_df['Currency'] == currency_list_el].tail(3000).copy()
                                            test = calculate_flag_trend_ma_simu(df=test, timeframe_fast=timeframe_fast_el,
                                                                                timeframe_slow=timeframe_slow_el,
                                                                                period_fast_small=period_fast_el,
                                                                                period_fast_big = period_fast_big_el,
                                                                                period_slow=period_slow_el,
                                                                                diff_pips=diff_pips_el,
                                                                                direction=direction_el)
                                            test = calculate_criterion_ma_simu(test, "", currency_list_el, 0.0010, sl_el,
                                                                               480)
                                            test = test.tail(2000).copy()
                                            test = test.head(1440).copy()
                                            result = result.append({'currency': currency_list_el,
                                                                    'time_min': test['time'].min(),
                                                                    'time_max': test['time'].max(),
                                                                    'timeframe_fast': timeframe_fast_el,
                                                                    'timeframe_slow': timeframe_slow_el,
                                                                    'period_fast_small': period_fast_el,
                                                                    'period_fast_big': period_fast_big_el,
                                                                    'period_slow': period_slow_el,
                                                                    'diff_pips': diff_pips_el,
                                                                    'direction': direction_el,
                                                                    'sl': sl_el,
                                                                    'profit_buy': test['profit_buy'].sum(),
                                                                    'profit_sell': test['profit_sell'].sum()},
                                                                   ignore_index=True)
                            else:
                                pass
        result.to_csv(f"{definitions.EXTERNAL_DIR}/ma_simu/result_{currency_list_el}.csv", index=False)
        # test_all.to_csv(f"{definitions.EXTERNAL_DIR}/ma_simu/test_{currency_list_el}.csv", index=False)

        best_buy_results = result[result['direction'] == 'buy'].copy()
        best_buy_results = best_buy_results[
            best_buy_results['profit_buy'] == best_buy_results['profit_buy'].max()].copy()
        best_sell_results = result[result['direction'] == 'sell'].copy()
        best_sell_results = best_sell_results[
            best_sell_results['profit_sell'] == best_sell_results['profit_sell'].max()].copy()
        ma_best_simu_results = ma_best_simu_results.append(best_buy_results, ignore_index=True)
        ma_best_simu_results = ma_best_simu_results.append(best_sell_results, ignore_index=True)
    ma_best_simu_results.drop_duplicates(subset=['currency', 'direction'], keep='first', inplace=True)
    ma_best_simu_results.to_csv(f"{definitions.EXTERNAL_DIR}/ma_simu/ma_best_simu_results.csv", index=False)
