from resspect.salt3_utils import get_distances

import time
import pandas as pd
import numpy as np
import os


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
subtype = 1

subsample = 'DDF'

# identification of photoids file
version = 1

# max number of light curves to fit per run
chunck_size = 19000       

# path to zenodo test metadata
test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'

output_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data3/' + subsample + '/'

# path to photoids output directory
photoids_dir = os.path.join(output_dir, 'SALT2_fit', types_names[code_plasticc], 'photoids')

# path to fitres output directory
fitres_dir = os.path.join(output_dir, 'SALT2_fit', types_names[code_plasticc], 'fitres')

# path to distances output directory
dist_dir = os.path.join(output_dir, 'SALT2_fit', types_names[code_plasticc], 'distances')

# path to PLAsTiCC SNANA files
SNANA_dir = '/media/RESSPECT/data/PLAsTiCC/SNANA/'

# this should be False if you want to used previous fits
# if True it will proceed with fit for all objects
# if False it will check master file to avoid fits that are already done
restart_master = True

###############################################################
# create directories

dirlist = [os.path.join(output_dir, 'SALT2_fit'),
           os.path.join(output_dir,'SALT2_fit', types_names[code_plasticc]),
           photoids_dir, fitres_dir, dist_dir]

for name in dirlist:
    if not os.path.isdir(name):
        os.makedirs(name)

# translate names
code_snana = SNANA_types[code_plasticc]
if not isinstance(code_snana, int):
    code_snana = SNANA_types[code_plasticc][subtype]
    
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
    if code_plasticc in [62, 42, 6]:
        type_flag1 = test_metadata['true_target'].values == code_plasticc
        type_flag2 = test_metadata['true_submodel'].values == subtype
        type_flag = np.logical_and(type_flag1, type_flag2)
    else:
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

    for start in range(0, data_out.shape[0], chunck_size):
        i = i + 1
        
        upper_lim = min(start + chunck_size, data_out.shape[0])
        data_chuncks[i] = data_out.iloc[start:upper_lim]
    
        if code_plasticc in [62, 42, 6]:
            data_chuncks[i].to_csv(os.path.join(photoids_dir, 
                                                sntype + '_' + subsample + \
                                                '_photoids_plasticc_' + \
                                               str(SNANA_types[code_plasticc][subtype]) + \
                                               '_' + str(i) + '.dat'), 
                                                index=False)
        else:
            data_chuncks[i].to_csv(os.path.join(photoids_dir, 
                                                sntype + '_' + subsample + \
                                                '_photoids_plasticc_' + str(i) + '.dat'), 
                                                index=False)


# perform SALT2 fit     
start_time = time.time()
    
print('*********  Fitting light curves   ******************')

if code_plasticc in [62, 42, 6]:
    fname = os.path.join(photoids_dir, sntype + '_' + subsample + '_photoids_plasticc_' + \
                         str(SNANA_types[code_plasticc][subtype]) + '_' + str(version) + '.dat')
    salt2mu_prefix = 'test_salt2mu_res_' +  str(SNANA_types[code_plasticc][subtype]) + '_'+ str(version)
    fitres_prefix='test_fitres_' +  str(SNANA_types[code_plasticc][subtype]) + '_' + str(version)
    salt3_outfile='salt3pipeinput_' +  str(SNANA_types[code_plasticc][subtype]) + '_' + str(version) + '.txt'
    master_fitres_name=os.path.join(fitres_dir, 'master_fitres_' +  str(SNANA_types[code_plasticc][subtype]) + '_'+ str(version) + '.fitres')
else:    
    fname = os.path.join(photoids_dir, sntype + '_' + subsample + '_photoids_plasticc_' + str(version) + '.dat')
    salt2mu_prefix='test_salt2mu_res_' + str(version)
    fitres_prefix='test_fitres_' + str(version)
    salt3_outfile='salt3pipeinput_' + str(version) + '.txt'
    master_fitres_name=os.path.join(fitres_dir, 'master_fitres_' + str(version) + '.fitres')

meta = pd.read_csv(fname, index_col=False)
codes = np.unique(meta['code'].values)




res = get_distances(fname,
                     data_prefix='LSST_' + subsample,
                     data_folder=SNANA_dir,            
                     select_modelnum=None,
                     salt2mu_prefix=salt2mu_prefix,
                     fitres_prefix=fitres_prefix,
                     maxsnnum=50000,
                     select_orig_sample=['test'],
                     salt3_outfile=salt3_outfile,
                     data_prefix_has_sntype=False,
                     master_fitres_name=master_fitres_name, 
                     append_master_fitres=True,
                     restart_master_fitres=restart_master)
    
if code_plasticc in [62, 42, 6]:
    res['distances'].to_csv(os.path.join(dist_dir,'mu_' + sntype + '_' + subsample + \
                            '_plasticc_' +  str(SNANA_types[code_plasticc][subtype]) + '_' + str(version) + '.dat'), index=False)
else:
    res['distances'].to_csv(os.path.join(dist_dir,'mu_' + sntype + '_' + subsample + \
                            '_plasticc_' + str(version) + '.dat'), index=False)
        
print("--- %s seconds ---" % (time.time() - start_time))
