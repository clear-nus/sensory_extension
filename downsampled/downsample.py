from functools import partial

import numpy as np
import pandas as pd


def downsample(dataset, frequency, time_past=0.05, time_future=0.25, time_interval=0.005):
    
    downsampler_neutouch = {
        'neutouch_tool_20'         : partial(_downsample_neutouch_tool, 20),
        'neutouch_tool_30'         : partial(_downsample_neutouch_tool, 30),
        'neutouch_tool_50'         : partial(_downsample_neutouch_tool, 50),
        'neutouch_handover_rod'    : partial(_downsample_neutouch_handover, 'rod'),
        'neutouch_handover_box'    : partial(_downsample_neutouch_handover, 'box'),
        'neutouch_handover_plate'  : partial(_downsample_neutouch_handover, 'plate'),
        'neutouch_food_apple'      : partial(_downsample_neutouch_food, 'apple'),
        'neutouch_food_banana'     : partial(_downsample_neutouch_food, 'banana'),
        'neutouch_food_empty'      : partial(_downsample_neutouch_food, 'empty'),
        'neutouch_food_pepper'     : partial(_downsample_neutouch_food, 'pepper'),
        'neutouch_food_tofu'       : partial(_downsample_neutouch_food, 'tofu'),
        'neutouch_food_water'      : partial(_downsample_neutouch_food, 'water'),
        'neutouch_food_watermelon' : partial(_downsample_neutouch_food, 'watermelon')
    }
        
    downsampler_biotac = {
        'biotac_tool_20'           : partial(_downsample_biotac_tool, 20),
        'biotac_tool_30'           : partial(_downsample_biotac_tool, 30),
        'biotac_tool_50'           : partial(_downsample_biotac_tool, 50),
        'biotac_handover_rod'      : partial(_downsample_biotac_handover, 'rod'),
        'biotac_handover_box'      : partial(_downsample_biotac_handover, 'box'),
        'biotac_handover_plate'    : partial(_downsample_biotac_handover, 'plate'),
        'biotac_food_apple'        : partial(_downsample_biotac_food, 'apple'),
        'biotac_food_banana'       : partial(_downsample_biotac_food, 'banana'),
        'biotac_food_empty'        : partial(_downsample_biotac_food, 'empty'),
        'biotac_food_pepper'       : partial(_downsample_biotac_food, 'pepper'),
        'biotac_food_tofu'         : partial(_downsample_biotac_food, 'tofu'),
        'biotac_food_water'        : partial(_downsample_biotac_food, 'water'),
        'biotac_food_watermelon'   : partial(_downsample_biotac_food, 'watermelon')
    }
    
    if dataset in downsampler_neutouch:
        signals, labels = downsampler_neutouch[dataset](time_past, time_future, time_interval, frequency)
        
    elif dataset in downsampler_biotac:
        signals, labels = downsampler_biotac[dataset](frequency)
        
    else: raise Exception('Dataset not found')
    
    np.savez(f'{dataset}_{frequency}hz.npz', signals=signals, labels=labels)
    print(f'Downsampling for {dataset} to {frequency} Hz completed with signals.shape = {signals.shape} and labels.shape = {labels.shape}')


#    _   _            _                   _        _____                                                _           
#   | \ | |          | |                 | |      |  __ \                                              | |          
#   |  \| | ___ _   _| |_ ___  _   _  ___| |__    | |  | | _____      ___ __  ___  __ _ _ __ ___  _ __ | | ___ _ __ 
#   | . ` |/ _ \ | | | __/ _ \| | | |/ __| '_ \   | |  | |/ _ \ \ /\ / / '_ \/ __|/ _` | '_ ` _ \| '_ \| |/ _ \ '__|
#   | |\  |  __/ |_| | || (_) | |_| | (__| | | |  | |__| | (_) \ V  V /| | | \__ \ (_| | | | | | | |_) | |  __/ |   
#   |_| \_|\___|\__,_|\__\___/ \__,_|\___|_| |_|  |_____/ \___/ \_/\_/ |_| |_|___/\__,_|_| |_| |_| .__/|_|\___|_|   
#                                                                                                | |                
        

def _downsample_neutouch_tool(tool_length, time_past, time_future, time_interval, frequency):
    
    signals = None
    labels = None
        
    for trial in range(1, 11):
        
        print('Downsampling trial {:02d} for neutouch_tool_{:02d}'.format(trial, tool_length))
        
        df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/tool_neutouch_1k/trial{trial}_{tool_length}_essentials.csv')
        df_raw = _read_neutouch_raw(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/tool_neutouch_1k/trial{trial}_{tool_length}.tact')
        
        signals_temp = _bin_neutouch_signal(df_essentials.t.values, df_raw, time_past, time_future, time_interval, frequency)
        labels_temp = df_essentials.label_y.values
        
        signals = np.append(signals, signals_temp, axis=0) if signals is not None else signals_temp
        labels = np.append(labels, labels_temp, axis=0) if labels is not None else labels_temp
    
    return signals, labels


def _downsample_neutouch_handover(item, time_past, time_future, time_interval, frequency):
    
    signals = None
    labels = None
    
    df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/neutouch_handover/nt_essentials.csv')
    df_essentials = df_essentials[df_essentials.obj == item]
    
    for _, row in df_essentials.iterrows():
        
        print(f'Downsampling {row.fname} for neutouch_handover_{row.obj}')
        
        df_raw = _read_neutouch_raw(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/neutouch_handover/{row.fname}.tact')
        tap_time = row.tapped_time
        
        signals_temp = _bin_neutouch_signal(np.array([tap_time]), df_raw, time_past, time_future, time_interval, frequency)
        labels_temp = row[['isPos', 'label_x_thumb', 'label_y_thumb', 'label_z_thumb', 'label_x_thumb_d', 'label_y_thumb_d', 'label_z_thumb_d', 'label_x_index', 'label_y_index', 'label_z_index', 'label_x_index_d', 'label_y_index_d', 'label_z_index_d']].values.astype('float')
        
        signals = np.vstack((signals, signals_temp)) if signals is not None else signals_temp
        labels = np.vstack((labels, labels_temp)) if labels is not None else labels_temp
    
    return signals, labels


def _downsample_neutouch_food(item, time_past, time_future, time_interval, frequency):
    
    import glob
    
    signals = None
    labels = None
    
    for filename in glob.glob(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/food/food_poking_batch[0-9]/{item}_zero_*[0-9].tact'):
        
        df_raw = _read_neutouch_raw(filename)
        start_time = df_raw['t'][0] + 2
        signal_temp = _bin_neutouch_signal(np.array([start_time]), df_raw, 0, 6, 0.05, frequency)
        signals = np.vstack((signals, signal_temp)) if signals is not None else signal_temp
    
    labels = np.ones(signals.shape[0])
    
    return signals, labels


def _read_neutouch_raw(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'removable', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(['removable'], axis=1, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    return df.reset_index(drop=True)


def _bin_neutouch_signal(tap_times, df_raw, time_past, time_future, time_interval, frequency):

    n_bins = int((time_past + time_future) / time_interval) + 1
    signals = np.zeros([len(tap_times), 80, n_bins], dtype=int)

    summer = 0
    vals = np.array([])
    
    for i, tap_time in enumerate(tap_times):
        
        df_timespan = df_raw[(df_raw.t >= (tap_time - time_past)) & (df_raw.t < (tap_time + time_future))]
        df_timespan = df_timespan.reset_index(drop=True)
        
        last_spiked = np.zeros(80)
        indices = []
        
        for j, sample in df_timespan.iterrows():
            
            if (sample.t - last_spiked[int(sample.taxel) - 1]) >= 1 / frequency:
                
                indices.append(j)
                last_spiked[int(sample.taxel) - 1] = sample.t
        
        df_timespan = df_timespan.iloc[indices]
        df_timespan = df_timespan.reset_index(drop=True)
        
        df_positive = df_timespan[df_timespan.isPos == 1]
        df_negative = df_timespan[df_timespan.isPos == 0]

        t = tap_time - time_past
        k = 0

        while t < (tap_time + time_future):
            
            positive_taxels = df_positive[((df_positive.t >= t) & (df_positive.t < t + time_interval))].taxel
            if len(positive_taxels):
                for taxel in positive_taxels:
                    signals[i, taxel - 1, k] += 1
                    
            negative_taxels = df_negative[((df_negative.t >= t) & (df_negative.t < t + time_interval))].taxel
            if len(negative_taxels):
                for taxel in negative_taxels:
                    signals[i, taxel - 1, k] -= 1
                    
            t += time_interval
            k += 1
    
    return signals


#    ____  _       _                _____                                                _           
#   |  _ \(_)     | |              |  __ \                                              | |          
#   | |_) |_  ___ | |_ __ _  ___   | |  | | _____      ___ __  ___  __ _ _ __ ___  _ __ | | ___ _ __ 
#   |  _ <| |/ _ \| __/ _` |/ __|  | |  | |/ _ \ \ /\ / / '_ \/ __|/ _` | '_ ` _ \| '_ \| |/ _ \ '__|
#   | |_) | | (_) | || (_| | (__   | |__| | (_) \ V  V /| | | \__ \ (_| | | | | | | |_) | |  __/ |   
#   |____/|_|\___/ \__\__,_|\___|  |_____/ \___/ \_/\_/ |_| |_|___/\__,_|_| |_| |_| .__/|_|\___|_|   
#                                                                                 | |                


def _downsample_biotac_signals(signals, frequency):

    samples_to_keep = int(frequency / 2200 * signals.shape[1]) + 1
    indices = np.round(np.linspace(0, signals.shape[1] - 1, samples_to_keep)).astype(int)
    
    return signals[:, indices]


def _downsample_biotac_tool(tool_length, frequency):
    
    npzfile = np.load(f'../preprocessed/biotac_tool_{tool_length}.npz')
    
    signals = npzfile['signals'] 
    labels = npzfile['labels']
    
    return _downsample_biotac_signals(signals, frequency), labels


def _downsample_biotac_handover(item, frequency):
    
    npzfile = np.load(f'../preprocessed/biotac_handover_{item}.npz')
    
    signals = npzfile['signals'] 
    labels = npzfile['labels']
    
    return _downsample_biotac_signals(signals, frequency), labels
    

def _downsample_biotac_food(item, frequency):
    
    npzfile = np.load(f'../preprocessed/biotac_food_{item}.npz')
    
    signals = npzfile['signals'] 
    labels = npzfile['labels']
    
    return _downsample_biotac_signals(signals, frequency), labels


#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) == 2:
        for frequency in [5, 50, 100, 250, 500, 1000, 2000, 3000]:
            downsample(sys.argv[1], frequency)
        
    if len(sys.argv) == 3:
        downsample(sys.argv[1], int(sys.argv[2]))
    