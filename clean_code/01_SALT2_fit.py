from resspect.salt3_utils import get_distances

import time
import pandas as pd
import numpy as np


#############################################################
########    Translate types SNANA/zendo    ##################

SNANA_types = {90:11, 62:{1:3, 2:13}, 42:{1:2, 2:12, 3:14},
               67:41, 52:43, 64:51, 95:60, 994:61, 992:62,
               993:63, 15:64, 88:70, 92:80, 65:81, 16:83,
               53:84, 991:90, 6:{1:91, 2:93}}

types_names = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
               95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 
               65:'M-dwarf', 16:'EB',53:'Mira', 6:'MicroL', 
               991:'MicroLB', 992:'ILOT', 993:'CART', 994:'PISN',
               995:'MLString'}

#############################################################
##########   User choices             #######################

# getting the photoids only needs to be done once
get_photoids = True

# SALT2 fit is separated by class
code_plasticc = 90

subsample = 'DDF'

# identification of photoids file
version = 1

# max number of light curves to fit per run
chunck_size = 15000          

# path to zenodo test metadata
test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'

# path to photoids output directory
photoids_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data/' + \
                subsample +'/SALT2_fit/' + types_names[code_plasticc] + '/photoids/'

# path to fitres output directory
fitres_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data/' + \
                subsample +'/SALT2_fit/' + types_names[code_plasticc] + '/fitres'

# path to distances output directory
fitres_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data/' + \
                subsample +'/SALT2_fit/' + types_names[code_plasticc] + '/distances/'

# path to PLAsTiCC SNANA files
SNANA_dir = '/media/RESSPECT/data/PLAsTiCC/SNANA/'

# this should be False if you want to used previous fits
# if True it will proceed with fit for all objects
# if False it will check master file to avoid fits that are already done
restart_master = True

###############################################################

# translate names
code_snana = SNANA_types[code_plasticc]
sntype = types_names[code_plasticc]

if get_photoids:
    
    # read plasticc test metadata
    test_metadata = pd.read_csv(test_zenodo_meta)

    # separate per observation strategy
    if subsample == 'DDF':
        ddf_flag = test_metadata['ddf_bool'].values == 1
    else:
        ddf_flag = test_metadata['ddf_bool'].values == 0
    
    # get only one class
    type_flag = test_metadata['true_target'].values == code_plasticc
    mask = np.logical_and(ddf_flag, type_flag)

    # get all objects of this class in the chosen strategy
    snids = test_metadata[mask]['object_id'].values
    redshifts = test_metadata[mask]['true_z'].values

    # prepare data for SALT2 fit
    data_new = {}
    data_new['id'] = snids
    data_new['redshift'] = redshifts
    data_new['type'] = [sntype for i in range(snids.shape[0])]
    data_new['code'] = [code_snana for i in range(snids.shape[0])]
    data_new['orig_sample'] = ['test' for i in range(snids.shape[0])]
    data_new['code_zenodo'] = [code_plasticc for i in range(snids.shape[0])]
        
    data_out = pd.DataFrame(data_new)


    # separate data in chuncks to avoid numerical problems
    i = 0
    data_chuncks = {}

    for start in range(0, data_out.shape[0], chunk_size):
        i = i + 1
        
        upper_lim = min(start + chunk_size, data_out.shape[0])
        data_chuncks[i] = data_out.iloc[start:upper_lim]
    
        subset.to_csv(photoids_dir + '/' + sntype + '_' + subsample + \
                     '_photoids_plasticc_' + str(i) + '.dat', index=False)


# perform SALT2 fit     
start_time = time.time()
    
print('*********  Fitting light curves   ******************')

fname = photoids_dir + '/' + sntype + '_' + subsample + \
       '_photoids_plasticc_' + str(version) + '.dat'

meta = pd.read_csv(fname, index_col=False)
codes = np.unique(meta['code'].values)

res = get_distances(fname,
                     data_prefix='LSST_' + subsample,
                     data_folder=SNANA_dir,            
                     select_modelnum=None,
                     salt2mu_prefix='test_salt2mu_res_' + str(version),
                     fitres_prefix='test_fitres_' + str(version),
                     maxsnnum=50000,
                     select_orig_sample=['test'],
                     salt3_outfile='salt3pipeinput_' + str(version) + '.txt',
                     data_prefix_has_sntype=False,
                     master_fitres_name=fitres_dir + '/master_fitres_' + str(version) + '.fitres', 
                     append_master_fitres=True,
                     restart_master_fitres=restart_master)
    
res['distances'].to_csv(dist_dir + 'mu_' + sntype + '_' + subsample + \
                        '_plasticc_' + str(version) + '.dat', index=False)
    
        
print("--- %s seconds ---" % (time.time() - start_time))
