#!/usr/bin/env python

import rosbag
from os import listdir
from os.path import isfile, join
import pickle
import glob
import pandas as pd
import numpy as np


data_dir = '/home/crslab/clear_lab_stable/hh_updated/updated_hh/demo_junk_results/'
save_dir = '/home/crslab/clear_lab_stable/hh_updated/updated_hh/demo_junk_results/processed/'

# get files
_bag_files = glob.glob(data_dir + '*.bag')

rm_files = ['original', 'calibrated', 'optitrack']
bag_files =[]
for bag_file in _bag_files:
	add = True
	for _rm in rm_files:
		if _rm  in bag_file:
			add = False
			break
	if add:
		bag_files.append(bag_file)



fname=[]
for bag_file in bag_files:
	fname.append( bag_file.split('/')[-1])


for bag_file in bag_files:
	fname= bag_file.split('/')[-1]
	print('doing ', fname, ' ...')

	bag = rosbag.Bag(bag_file)


	data = []
	for topic, msg, t in bag.read_messages(['franka_state_controller']):
		t = msg.header.stamp.secs+msg.header.stamp.nsecs*1e-9
		W = msg.O_F_ext_hat_K
		data.append([t, W[0], W[1], W[2], W[3], W[4], W[5], msg.elbow[0]])
	bag.close()
	w_df = pd.DataFrame(data=data, columns=['t', 'x', 'y','z','xm','ym','zm', 'elbow'])
	w_df.to_csv(save_dir + fname[:-4] + '.csv', index=False)

	#pickle.dump(data, open(save_dir + fname[:-4] + '.pkl', 'wb'))

