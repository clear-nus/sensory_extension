import pandas as pd
import numpy as np
# from utils import read_neutouch_adc
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

fname = 'apple_zero_1111.tact'

def read_neutouch_adc(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'adc', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    df.t -= df.t.iloc[0]
    
    return df.reset_index(drop=True)

def filter_taxels(sample, threshold=300):
    mask = (sample.adc > -100)
    for taxel in range(1,81):
        subsample = df[sample.taxel == taxel]
        if subsample.shape[0] > 0:
            if subsample.adc.iloc[0] >= threshold:
                mask &= ~(sample.taxel==taxel)
    return sample[mask]


######
# can I detect the water?

def isApple(sample, threshold_unique_taxels=15, threshold_time=3.9):
    
    a = sample[sample.t >= threshold_time]

    if len(a.taxel.unique()) >= threshold_unique_taxels:
        return 0
    
    
    return 1

def isWater(sample, threshold_unique_taxels=4, threshold_diff_adc=4):

    a = sample[['taxel', 'adc']].groupby(['taxel']).agg(['max', 'min', 'std'])
    a.columns = a.columns.map('_'.join)
    a = a.assign(diversion = a.adc_max.subtract(a.adc_min))

    
    if ( a[a.diversion >= threshold_diff_adc].shape[0] ) <= threshold_unique_taxels:
        return 1
        
    return 0

# can I detect the tofu?

def isTofu(sample, threshold_unique_taxels=3, threshold_diff_adc=15):
    
    if isWater(sample) == 1:
        return 0
    if isApple(sample) == 1:
        return 0

    a = sample[['taxel', 'adc']].groupby(['taxel']).agg(['max', 'min', 'std'])
    a.columns = a.columns.map('_'.join)
    a = a.assign(diversion = a.adc_max.subtract(a.adc_min))

    print(a[a.diversion >= threshold_diff_adc].shape[0], threshold_unique_taxels)

    
    if a[a.diversion >= threshold_diff_adc].shape[0] >= threshold_unique_taxels:
        return 0
    
    return 1

# can I detect watermelon?

def integrate(sample):
    delta_times = sample.t.diff().fillna(0).values
    adc = sample.adc.values
    res = delta_times*adc
    return np.sum(res)

def isWatermelon(sample, threshold_unique_taxels=15, threshold_time=3.9):
    
    if isWater(sample) == 1:
        return 0
    
    if isTofu(sample) == 1:
        return 0
    
    a = sample[sample.t >= threshold_time]
    
    if len(a.taxel.unique()) >= threshold_unique_taxels:
        return 1
    
    return 0


### infer

df = read_neutouch_adc(fname)
df = df[(df.taxel >= 1) & (df.taxel <=80)]
df = filter_taxels(df)

print('water: ', isWater(df))
print('watermelon: ', isWatermelon(df))
print('tofu: ', isTofu(df))
print('apple: ', isApple(df))