# can I detect the water?

def isCucumber(sample, threshold_unique_taxels=15, threshold_time=3.9):
    
    a = sample[sample.t >= threshold_time]

    if len(a.taxel.unique()) >= threshold_unique_taxels:
        return 0
    
    return 1

# can I detect the tofu?

def isTofu(sample, threshold_unique_taxels=6, threshold_diff_adc=12):
    
    if isCucumber(sample) == 1:
        return 0

    a = sample[['taxel', 'adc']].groupby(['taxel']).agg(['max', 'min', 'std'])
    a.columns = a.columns.map('_'.join)
    a = a.assign(diversion = a.adc_max.subtract(a.adc_min))

    #print(a[a.diversion >= threshold_diff_adc].shape[0], threshold_unique_taxels)

    
    if a[a.diversion >= threshold_diff_adc].shape[0] >= threshold_unique_taxels:
        return 0
    
    return 1

# can I detect pressed tofu?

def isPressedTofu(sample, threshold_unique_taxels=8, threshold_diff_adc=20):
    if isCucumber(sample) == 1:
        return 0
    if isTofu(sample) == 1:
        return 0
    
    a = sample[['taxel', 'adc']].groupby(['taxel']).agg(['max', 'min', 'std'])
    a.columns = a.columns.map('_'.join)
    a = a.assign(diversion = a.adc_max.subtract(a.adc_min))
    
    if a[a.diversion >= threshold_diff_adc].shape[0] >= threshold_unique_taxels:
        return 0
    
    return 1
    
    

def integrate(sample):
    delta_times = sample.t.diff().fillna(0).values
    adc = sample.adc.values
    res = delta_times*adc
    return np.sum(res)

def isWatermelon(sample, threshold_unique_taxels=15, threshold_time=3.9):
    
    if isCucumber(sample) == 1:
        return 0
    
    if isPressedTofu(sample) == 1:
        return 0
    
    if isTofu(sample) == 1:
        return 0
    
    return 1