import pandas as pd
import numpy as np
import pystan
import os
import pickle


# translate between SNANA and zenodo ids
SNANA_types = {90:11, 62:{1:3, 2:13}, 42:{1:2, 2:12, 3:14},
               67:41, 52:43, 64:51, 95:60, 994:61, 992:62,
               993:63, 15:64, 88:70, 92:80, 65:81, 16:83,
               53:84, 991:90, 6:{1:91, 2:93}}

types_names = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
               95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
               16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
               993:'CART', 994:'PISN',995:'MLString'}


##################    user choices     ##########################################
case = '99.9SNIa0.1AGN'                         # choose contamination case
nbins = 30                                    # number of bins for SALT2mu
om_pri = [0.3, 0.01]                          # gaussian prior on om => [mean, std]
w_pri = [-10, 9]                              # flat prior on w
lowz = True                                   # choose to add lowz sample
field = 'DDF'                                 # choose field
version = '0'                                 # realization or version

# path to input (auxiliary files) and output directories
output_root = '/media2/RESSPECT2/clean_output/'
dir_output = output_root + field + '/v' + version + '/'

dir_input = output_root + '/misc/'

# path to input files in the coin01 server
fname_test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'
fname_sample = dir_output + 'samples/' + case + '.csv'
fname_salt2mu = 'SALT2mu.input'
fname_fitres_lowz = dir_input + 'lowz_only_fittres.fitres'
###################################################################################

#######################
### Generate fitres ###
#######################

# read plasticc test metadata
test_metadata = pd.read_csv(fname_test_zenodo_meta)

# read sample for this case
fname = dir_output + '/samples/' + case + '.csv'
fitres_main = pd.read_csv(fname, delim_whitespace=True, index_col=False)

# read lowz sample
if lowz:
    fitres_lowz = pd.read_csv(fname_fitres_lowz, index_col=False, comment="#", 
                              skip_blank_lines=True, delim_whitespace=True)
    fitres_lowz['zHD'] = fitres_lowz['SIM_ZCMB']
    fitres_lowz.fillna(value=-99, inplace=True)
    fitres = pd.concat([fitres_lowz, fitres_main[list(fitres_lowz.keys())]], 
                       ignore_index=True)
    
else:
    fitres = fitres_main 
    
# save combined fitres to file
fitres.to_csv(dir_output + 'fitres/master_fitres_' + case + '_lowz_withbias.fitres', 
              sep=" ", index=False)

#######################
####### SALT2mu #######
#######################

dir_list = ['results/',
            'results/' + field,
            'results/' + field + '/SALT2mu/']

for name in dir_list:
    if not os.path.isdir(name):
        os.makedirs(name)

# SALT2mu input file name
salt2mu_fname_input = dir_input + '/SALT2mu.input'
salt2mu_fname_output = 'results/' + field + '/SALT2mu/SALT2mu_' + case + '.input'

# change parameters for SALT2mu
op = open(salt2mu_fname_input, 'r')
lin = op.readlines()
op.close()

lin[0] = 'bins=' + str(nbins) + '\n'
lin[-3] = 'prefix=' + dir_output + '/test_salt2mu_lowz_withbias_' + case + '\n'
lin[-4] = 'file=' + dir_output + 'fitres/master_fitres_' + case + '_lowz_withbias.fitres' + '\n'

# store ourput file name
fname_fitres_comb = dir_output + 'fitres/test_salt2mu_lowz_withbias_' + case + '.fitres'

op2 = open(salt2mu_fname_output, 'w')
for line in lin:
    op2.write(line)
op2.close()

# get distances from SALT2MU
os.system('SALT2mu.exe ' + salt2mu_fname_output)

# move files
os.replace(dir_output + '/test_salt2mu_lowz_withbias_' + case + '.fitres',
           dir_output + '/fitres/test_salt2mu_lowz_withbias_' + case + '.fitres')
os.replace(dir_output + '/test_salt2mu_lowz_withbias_' + case + '.M0DIF',
          dir_output + '/M0DIF/test_salt2mu_lowz_withbias_' + case + '.M0DIF')
os.replace(dir_output + '/test_salt2mu_lowz_withbias_' + case + '.COV',
          dir_output + '/COV/test_salt2mu_lowz_withbias_' + case + '.COV')


#######################
#######   wfit  #######
#######################

# get current location
dir_work = os.getcwd()

# change working directory
os.chdir(dir_output + 'M0DIF/')

# run wfit
os.system('wfit.exe test_salt2mu_lowz_withbias_' + case + '.M0DIF -ompri ' + \
          str(om_pri[0]) + ' -dompri ' + str(om_pri[1]) + \
          ' -hmin 70 -hmax 70 -hsteps 1 -wmin -10 -wmax 9 -ommin ' + \
          str(om_pri[0] - om_pri[1]) + ' -ommax ' + str(om_pri[0] + om_pri[1]))

# move output file
os.rename(dir_output + 'M0DIF/test_salt2mu_lowz_withbias_' + case + '.M0DIF.cospar', 
          dir_output + 'cospar/test_salt2mu_lowz_withbias_' + case + '.M0DIF.cospar')

# go back to working directory
os.chdir(dir_work)

#######################
##### Stan model ######
#######################

# read data for Bayesian model
fitres_final = pd.read_csv(fname_fitres_comb, index_col=False, comment="#", 
                          skip_blank_lines=True, delim_whitespace=True)

# set initial conditions
z0 = 0
E0 = 0
c = 3e5
H0 = 70

# add small offset to duplicate redshifts
z_all = []
for j in range(fitres_final.shape[0]):
    z = fitres_final.iloc[j]['SIM_ZCMB']
    z_new = z
    if z in z_all:
        while z_new in z_all:
            z_new = z + np.random.normal(loc=0, scale=0.001)
            
    fitres_final.at[j, 'SIM_ZCMB'] = z_new
    z_all.append(z_new)

# order data according to redshift 
indx = np.argsort(fitres_final['SIM_ZCMB'].values)

# create input for stan model
stan_input = {}
stan_input['nobs'] = fitres_final.shape[0]
stan_input['z'] = fitres_final['SIM_ZCMB'].values[indx]
stan_input['mu'] = fitres_final['MU'].values[indx]
stan_input['muerr'] = fitres_final['MUERR'].values[indx]
stan_input['z0'] = z0
stan_input['H0'] = H0
stan_input['c'] = c
stan_input['E0'] = np.array([E0])
stan_input['ompri'] = om_pri[0]
stan_input['dompri'] = om_pri[1]
stan_input['wmin'] = w_pri[0]
stan_input['wmax'] = w_pri[1]

# save only stan input to file
fname_stan_input = dir_output + 'stan_input/stan_input_salt2mu_lowz_withbias_' + case + '.csv'

stan_input2 = {}
stan_input2['z'] = stan_input['z']
stan_input2['mu'] = stan_input['mu']
stan_input2['muerr'] = stan_input['muerr']
stan_input_tofile = pd.DataFrame(stan_input2)
stan_input_tofile.to_csv(fname_stan_input, index=False)

# fit Bayesian model
model = pystan.StanModel(file = dir_input + '/cosmo.stan')
fit = model.sampling(data=stan_input, iter=12000, chains=5, 
                     warmup=10000, control={'adapt_delta':0.99})

# get summary
res = fit.stansummary(pars=["om", "w"])
check = str(pystan.check_hmc_diagnostics(fit))




