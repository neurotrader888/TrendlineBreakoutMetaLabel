import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single, fit_upper_trendline
import mplfinance as mpf



def trendline_breakout_dataset(
        ohlcv: pd.DataFrame, lookback: int, 
        hold_period:int=12, tp_mult: float=3.0, sl_mult: float=3.0, 
        atr_lookback: int=168
):
    assert(atr_lookback >= lookback)

    close = np.log(ohlcv['close'].to_numpy())
   
    # ATR for normalizing, setting stop loss take profit
    atr = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']), np.log(ohlcv['close']), atr_lookback)
    atr_arr = atr.to_numpy()
   
    # Normalized volume
    vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()).to_numpy() 
    adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    trades = pd.DataFrame()
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    for i in range(atr_lookback, len(ohlcv)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line
        r_val = r_coefs[1] + lookback * r_coefs[0]

        # Entry
        if not in_trade and close[i] > r_val:
            
            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr'] = atr_arr[i]
            trades.loc[trade_i, 'sl'] = sl_price 
            trades.loc[trade_i, 'tp'] = tp_price 
            trades.loc[trade_i, 'hp_i'] = i + hold_period
            
            trades.loc[trade_i, 'slope'] = r_coefs[0]
            trades.loc[trade_i, 'intercept'] = r_coefs[1]


            # Trendline features
            # Resist slope
            trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i] 
       
            # Resist erorr
            line_vals = (r_coefs[1] + np.arange(lookback) * r_coefs[0])
            err = np.sum(line_vals  - window ) / lookback
            err /= atr_arr[i]
            trades.loc[trade_i, 'tl_err'] = err

            # Max distance from resist
            diff = line_vals - window
            trades.loc[trade_i, 'max_dist'] = diff.max() / atr_arr[i]

            # Volume on breakout
            trades.loc[trade_i, 'vol'] = vol_arr[i]

            # ADX
            trades.loc[trade_i, 'adx'] = adx_arr[i]


        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = close[i]
                
                in_trade = False
                trade_i += 1

    trades['return'] = trades['exit_p'] - trades['entry_p']
    
    # Features
    data_x = trades[['resist_s', 'tl_err', 'vol', 'max_dist', 'adx']]
    # Label
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1

    return trades, data_x, data_y

if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    trades, data_x, data_y = trendline_breakout_dataset(data, 72)

    # Drop any incomplete trades
    trades = trades.dropna()

    # Look at trades without any ML filter. 
    signal = np.zeros(len(data))
    for i in range(len(trades)):
        trade = trades.iloc[i]
        signal[int(trade['entry_i']):int(trade['exit_i'])] = 1.

    data['r'] = np.log(data['close']).diff().shift(-1)
    data['sig'] = signal
    returns = data['r'] * data['sig']
    
    print("Profit Factor", returns[returns > 0].sum() / returns[returns < 0].abs().sum())
    print("Win Rate", len(trades[trades['return'] > 0]) / len(trades))
    print("Average Trade", trades['return'].mean()) 


    plt.style.use('dark_background')
    returns.cumsum().plot()
    plt.ylabel("Cumulative Log Return")





