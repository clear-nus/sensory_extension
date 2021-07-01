# Waits for filename node to be published.
# Keep an interval of ___ seconds 

import pandas as pd
import numpy as np
import rospy
from std_msgs.msg import String
# from utils import read_neutouch_adc
import matplotlib.pyplot as plt
import os
import time

pd.options.mode.chained_assignment = None  # default='warn'

# fname = 'apple_zero_0.tact'

def filename_callback(msg):
    global df 
    time.sleep(10)
    filename = '/home/crslab/clear_lab_stable/hh_updated/updated_hh/demo_junk_results/' + msg.data + '.tact'
    if os.path.exists(filename):
        print('file found.', filename)
        df = read_neutouch_adc(filename)
        df = df[(df.taxel >= 1) & (df.taxel <=80)]
        df = filter_taxels(df)
        print('here')
        if isWater(df):
            infer_pub.publish('water')
        elif isWatermelon(df):
            infer_pub.publish('watermelon')
        elif isTofu(df):
            infer_pub.publish('tofu')
        elif isApple(df):
            infer_pub.publish('apple')


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
    global df
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

def isWater(sample, threshold_unique_taxels=5, threshold_diff_adc=5 ):

    a = sample[['taxel', 'adc']].groupby(['taxel']).agg(['max', 'min', 'std'])
    a.columns = a.columns.map('_'.join)
    a = a.assign(diversion = a.adc_max.subtract(a.adc_min))

    
    if ( a[a.diversion >= threshold_diff_adc].shape[0] ) <= threshold_unique_taxels:
        return 1
        
    return 0

# can I detect the tofu?

def isTofu(sample, threshold_unique_taxels=6, threshold_diff_adc=15):
    
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

# df = read_neutouch_adc(fname)
# df = df[(df.taxel >= 1) & (df.taxel <=80)]
# df = filter_taxels(df)

# print('water: ', isWater(df))
# print('watermelon: ', isWatermelon(df))
# print('tofu: ', isTofu(df))
# print('apple: ', isApple(df))
if __name__ == '__main__':
    rospy.init_node('infer_node')
    filename_sub = rospy.Subscriber('filename', String, filename_callback)
    infer_pub    = rospy.Publisher('output', String, queue_size =10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass