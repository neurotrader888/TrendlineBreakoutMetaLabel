import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from trendline_break_dataset import trendline_breakout_dataset
from sklearn.ensemble import RandomForestClassifier


def walkforward_model(
        close: np.array, trades:pd.DataFrame, 
        data_x: pd.DataFrame, data_y: pd.Series, 
        train_size: int, step_size: int
):
    
    signal = np.zeros(len(close))
    prob_signal = np.zeros(len(close))

    next_train = train_size
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
      

    model = None 
    for i in range( len(close)):
        if i == next_train:
            start_i = i - train_size

            train_indices = trades[(trades['entry_i'] > start_i) & (trades['exit_i'] < i)].index

            x_train = data_x.loc[train_indices]
            y_train = data_y.loc[train_indices]
            print('training', i, 'N cases', len(train_indices))
            model = RandomForestClassifier(n_estimators=1000, max_depth=3, random_state=69420)
            model.fit(x_train.to_numpy(), y_train.to_numpy())

            next_train += step_size
        
        if in_trade:
            if close[i] >= tp_price or close[i] <= sl_price or i >= hp_i:
                signal[i] = 0
                prob_signal[i] = 0
                in_trade = False
            else:
                signal[i] = signal[i - 1]
                prob_signal[i] = prob_signal[i - 1]


        if  trade_i < len(trades) and i == trades['entry_i'].iloc[trade_i]:
            
            if model is not None:
                prob = model.predict_proba(data_x.iloc[trade_i].to_numpy().reshape(1, -1))[0][1]
                prob_signal[i] = prob

                trades.loc[trade_i, 'model_prob'] = prob

                if prob > 0.5: # greater than 50%, take trade
                    signal[i] = 1
                
                in_trade = True
                trade = trades.iloc[trade_i]
                tp_price = trade['tp'] 
                sl_price = trade['sl'] 
                hp_i = trade['hp_i'] 

            trade_i += 1

    return signal, prob_signal 





if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = data.dropna()

    trades, data_x, data_y = trendline_breakout_dataset(data, 72)
    signal, prob = walkforward_model(
            np.log(data['close']).to_numpy(), 
            trades, data_x, data_y, 
            365 * 24 * 2, 365 * 24
    )
    
    data['sig'] = signal

    # dumb_sig takes every trade, no ML filter
    data['dumb_sig'] = prob
    data.loc[data['dumb_sig'] > 0, 'dumb_sig'] = 1
   
    data = data[data.index > '2020-01-01']
    data['r'] = np.log(data['close']).diff().shift(-1)


    # Compute trade stats for all trades vs. model's selected trades
    trades = trades.dropna() 
    all_r = trades['return']    
    mod_r = trades[trades['model_prob'] > 0.5]['return']

    no_filter_rets = data['r'] * data['dumb_sig']
    filter_rets = data['r'] * data['sig']

    def prof_factor(rets):
        return rets[rets>0].sum() / rets[rets<0].abs().sum()
    
    
    print("All Trades PF", prof_factor(no_filter_rets))
    print("All Trades Avg", all_r.mean())
    print("All Trades Win Rate", len(all_r[all_r > 0]) / len(all_r) )
    print("All Trades Time In Market", len(data[data['dumb_sig'] > 0]) / len(data) )
    
    print("Meta-Label Trades PF", prof_factor(filter_rets)) 
    print("Meta-Label Trades Avg", mod_r.mean())
    print("Meta-Label Trades Win Rate", len(mod_r[mod_r > 0]) / len(mod_r))
    print("Meta-Label Time In Market", len(data[data['sig'] > 0]) / len(data))
    
    plt.style.use('dark_background') 
    (data['r'] * data['sig']).cumsum().plot(label='Meta-Labeled')
    (data['r'] * data['dumb_sig']).cumsum().plot(label='All Trades')
    (data['r']).cumsum().plot(label='Buy Hold')
    plt.legend()
    plt.show()
    



