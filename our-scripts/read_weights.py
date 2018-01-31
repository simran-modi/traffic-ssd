#python 3

import numpy as np
import pickle

wt = np.load("weights.npy", encoding = 'latin1') #return 0-d numpy array containing a dictionary
wt_dict = wt.item()

for key in wt_dict.keys():
	print(key)
	for k in wt_dict[key].keys():
		print('\t'+k)


