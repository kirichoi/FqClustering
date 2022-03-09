# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:27:14 2021

@author: user
"""

import os
import numpy as np
import pandas as pd
from scipy import spatial
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import itertools
import ctypes

def formfactor(args):
    gen = np.genfromtxt(args)
    
    df = pd.DataFrame(gen)
    df.iloc[:,0] = df.iloc[:, 0].astype(int)
    df.iloc[:,1] = df.iloc[:, 1].astype(int)
    df.iloc[:,6] = df.iloc[:, 6].astype(int)
    
    morph_coor = np.divide(np.array(df[[2,3,4]]), 1000).tolist()
    ucoor = np.unique(morph_coor, axis=0)
    lenucoor = len(ucoor)
    
    q_range = np.logspace(-3,3,601)
    Pq_ALLEN = np.empty(len(q_range))
    ccdist = spatial.distance.cdist(ucoor, ucoor)
    ccdisttri = ccdist[np.triu_indices_from(ccdist, k=1)]
    
    for q in range(len(q_range)):
        qrvec = q_range[q]*ccdisttri
        Pq_ALLEN[q] = np.divide(np.divide(2*np.sum(np.sin(qrvec)/qrvec), lenucoor), lenucoor) + 1/lenucoor
    
    np.save(r'./pq/' + os.path.basename(args)[:-4] + '.npy', Pq_ALLEN)

if __name__ == '__main__': 
    PATH = r'./Skels connectome_mod'

    fp = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
    fp = [os.path.join(PATH, f) for f in fp]
    fp.sort()
    
	# multiprocessing 
	# use 30 threads and refresh the worker after 10 tasks
    pool = mp.Pool(30, maxtasksperchild=10)
    
    t1 = time.time()
    
    pool.map(formfactor, fp)
    
    t2 = time.time()
    
    print(t2-t1)
    
    time.sleep(5)
    
    pool.close()
    pool.join()
    
    time.sleep(5)



