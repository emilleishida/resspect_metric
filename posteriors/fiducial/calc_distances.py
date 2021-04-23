import os
from resspect.salt3_utils import get_distances
import pandas as pd
import numpy as np
import time

nloops = 1


res = {}

for n in range(nloops):
    
    start_time = time.time()
    
    print('********* Loop ' + str(n) + '   ******************')

    fname = 'fiducial_photoids_plasticc.dat'
    
    meta = pd.read_csv(fname, index_col=False)
    codes = np.unique(meta['code'].values)

    restart_master = True

    res[n] = get_distances(fname,
                                   data_prefix='LSST_DDF',
                                   data_folder='/media/RESSPECT/data/PLAsTiCC/SNANA',            
                                   select_modelnum=None,
                                   salt2mu_prefix='test_salt2mu_res',
                                   maxsnnum=50000,
                                   select_orig_sample=['test'],
                                   salt3_outfile='salt3pipeinput.txt',
                                   data_prefix_has_sntype=False,
                                   master_fitres_name='master_fitres.fitres',
                                   append_master_fitres=True,
                                   restart_master_fitres=restart_master)
    
    res[n]['distances'].to_csv('results/mu_photoIa_plasticc_perfect.dat', index=False)
    res[n]['cosmopars'].to_csv('results/cosmo_photoIa_plasticc_perfect.dat', index=False)
    
        
    print("--- %s seconds ---" % (time.time() - start_time))
