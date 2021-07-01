import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

obj = 'empty'
iteration = 11
a = pd.read_csv('./processed/{}_zero_{}.csv'.format(obj, iteration))
fig, axs = plt.subplots(7)
axs[0].plot(a.t, a.x)
axs[1].plot(a.t, a.y)
axs[2].plot(a.t, a.z)
axs[3].plot(a.t, a.xm)
axs[4].plot(a.t, a.ym)
axs[5].plot(a.t, a.zm)
axs[6].plot(a.t, a.elbow)
plt.show()