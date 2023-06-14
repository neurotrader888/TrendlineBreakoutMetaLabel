import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single
import mplfinance as mpf

def trendline_breakout(close: np.array, lookback:int):
    s_tl = np.zeros(len(close))
    s_tl[:] = np.nan

    r_tl = np.zeros(len(close))
    r_tl[:] = np.nan

    sig = np.zeros(len(close))

    for i in range(lookback, len(close)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line, projected forward to current bar
        s_val = s_coefs[1] + lookback * s_coefs[0] 
        r_val = r_coefs[1] + lookback * r_coefs[0] 

        s_tl[i] = s_val
        r_tl[i] = r_val

        if close[i] > r_val:
            sig[i] = 1.0
        elif close[i] < s_val:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1]

    return s_tl, r_tl, sig


if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()
    
    lookback = 72
    support, resist, signal = trendline_breakout(data['close'].to_numpy(), lookback)
    data['support'] = support
    data['resist'] = resist
    data['signal'] = signal

    plt.style.use('dark_background')
    data['close'].plot(label='Close')
    data['resist'].plot(label='Resistance', color='green')
    data['support'].plot(label='Support', color='red')
    plt.show()

    data['r'] = np.log(data['close']).diff().shift(-1)
    strat_r = data['signal'] * data['r']

    pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum() 
    print("Profit Factor", lookback,  pf)

    strat_r.cumsum().plot()
    plt.ylabel("Cumulative Log Return")
    plt.show()
    
    '''
    lookbacks = list(range(24, 169, 2))
    pfs = []

    lookback_returns = pd.DataFrame()
    for lookback in lookbacks:
        support, resist, signal = trendline_breakout(data['close'].to_numpy(), lookback)
        data['signal'] = signal

        data['r'] = np.log(data['close']).diff().shift(-1)
        strat_r = data['signal'] * data['r']

        pf = strat_r[strat_r > 0].sum() / strat_r[strat_r < 0].abs().sum() 
        print("Profit Factor", lookback,  pf)
        pfs.append(pf)

        lookback_returns[lookback] = strat_r

    plt.style.use('dark_background')
    x = pd.Series(pfs, index=lookbacks)
    x.plot()
    plt.ylabel("Profit Factor")
    plt.xlabel("Trendline Lookback")
    plt.axhline(1.0, color='white')

    '''







