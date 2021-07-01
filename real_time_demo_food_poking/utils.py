import pandas as pd
import numpy as np
import pickle

gs_estimator = None
model_loaded = False

def load_model(tool_length=20):
    global gs_estimator, model_loaded
    gs_estimator = pickle.load(open('tool' + str(tool_length) +'_fft_svmrbf.pkl', 'rb'))
    model_loaded = True
    
    
def read_neutouch_raw(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'removable', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(['removable'], axis=1, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    return df.reset_index(drop=True)

def read_neutouch_adc(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'adc', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    df.t -= df.t.iloc[0]
    
    return df.reset_index(drop=True)

def bin_neutouch_signal(tap_time, df_raw, time_past, time_future, time_interval):

    n_bins = int((time_past + time_future) / time_interval) + 1
    signal = np.zeros([1, 80, n_bins], dtype=int)


    df_timespan = df_raw[(df_raw.t >= (tap_time - time_past)) & (df_raw.t < (tap_time + time_future))]
    df_positive = df_timespan[df_timespan.isPos == 1]
    df_negative = df_timespan[df_timespan.isPos == 0]

    t = tap_time - time_past
    k = 0

    while t < (tap_time + time_future):

        positive_taxels = df_positive[((df_positive.t >= t) & (df_positive.t < t + time_interval))].taxel
        if len(positive_taxels):
            for taxel in positive_taxels:
                signal[0, taxel - 1, k] += 1

        negative_taxels = df_negative[((df_negative.t >= t) & (df_negative.t < t + time_interval))].taxel
        if len(negative_taxels):
            for taxel in negative_taxels:
                signal[0, taxel - 1, k] -= 1

        t += time_interval
        k += 1
    
    return signal, df_timespan

def infer(X, tool_length=20):
    global gs_estimator
    if gs_estimator == None:
        load_model(tool_length)
    X = X / 1000
    X = np.abs(np.fft.fft(X)) / 10
    X = np.reshape(X, (X.shape[0], -1))
    y = gs_estimator.predict(X)
    return y, X