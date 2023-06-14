import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from trendline_break_dataset import trendline_breakout_dataset
from trendline_automation import fit_trendlines_single
from sklearn.ensemble import RandomForestClassifier
import sklearn

def plot_trade(ohlc: pd.DataFrame, trades: pd.DataFrame, trade_i: int, lookback: int):
    plt.style.use('dark_background')

    trade = trades.iloc[trade_i]
    entry_i = int(trade['entry_i'])
    exit_i = int(trade['exit_i']) 
    
    candles = np.log(ohlc.iloc[entry_i - lookback:exit_i+1])
    resist = [(candles.index[0], trade['intercept']), (candles.index[lookback], trade['intercept'] + trade['slope'] * lookback)]
    tp = [(candles.index[lookback], trade['tp']), (candles.index[-1], trade['tp'])]
    sl = [(candles.index[lookback], trade['sl']), (candles.index[-1], trade['sl'])]

    mco = [None] * len(candles)
    mco[lookback] = 'blue'
    fig, axs = plt.subplots(2, sharex=True, height_ratios=[3, 1])
    axs[1].set_title('Volume')

    mpf.plot(candles, volume=axs[1], alines=dict(alines=[resist, tp, sl], colors=['w', 'b', 'r']), type='candle', style='charles', ax=axs[0], marketcolor_overrides=mco)
    plt.show()


data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
data = data.dropna()
data = data[data.index < '2019-01-01']

plt.style.use('dark_background')

trades, data_x, data_y = trendline_breakout_dataset(data, 72)
trades.plot.scatter('resist_s', 'return')
plt.show()



#plt.style.use('dark_background')
#ax = plt.gca()
#model = RandomForestClassifier(max_depth=2)
#model.fit(data_x, data_y)
#sklearn.tree.plot_tree(model.estimators_[0], feature_names=data_x.columns, ax=ax)







