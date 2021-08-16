case = 'fiducial'

import pandas as pd
import numpy as np
import pystan
import os
from resspect.salt3_utils import get_distances
import pickle
import time
from shutil import copyfile



fit_lightcurves = False
restart_master = True

# number of bins for SALT2mu
nbins = 70

# rather to re-write fitres file
replace_z = True
add_lowz  = True
bias = True

###########################################################################################
# translate ids                                         ###################################
###########################################################################################
SNANA_types = {90:11, 62:{1:3, 2:13}, 42:{1:2, 2:12, 3:14},
               67:41, 52:43, 64:51, 95:60, 994:61, 992:62,
               993:63, 15:64, 88:70, 92:80, 65:81, 16:83,
               53:84, 991:90, 6:{1:91, 2:93}}

types_names = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
               95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
               16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
               993:'CART', 994:'PISN',995:'MLString'}


# read plasticc test metadata
test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'
test_metadata = pd.read_csv(test_zenodo_meta)

# read sample for this case
fname = '/media/RESSPECT/data/PLAsTiCC/for_metrics/' + case + '_samp.csv'
data = pd.read_csv(fname)

data_new = {}
data_new['id'] = data['id'].values
data_new['redshift'] = data['redshift'].values
data_new['type'] = [types_names[item] for item in data['code'].values]
data_new['code'] = []
data_new['orig_sample'] = ['test' for i in range(data.shape[0])]
data_new['queryable'] = [True for i in range(data.shape[0])]
data_new['code_zenodo'] = data['code'].values

for i in range(data.shape[0]):            
    sncode = data.iloc[i]['code']
    if  sncode not in [62, 42, 6]:
        data_new['code'].append(SNANA_types[sncode])
        if SNANA_types[sncode] == 60:
            print('sncode = ', sncode, ' new code=', SNANA_types[sncode])
    else:
        flag = test_metadata['object_id'].values == data.iloc[i]['id']
        submodel = test_metadata[flag]['true_submodel'].values[0]
        data_new['code'].append(SNANA_types[sncode][submodel])
        
data_out = pd.DataFrame(data_new)
data_out.to_csv('results/' + case + '_photoids_plasticc.dat', index=False)

###################################################################################
###################################################################################


res = {}

if fit_lightcurves:
    
    start_time = time.time()
    
    print('*********  Fitting light curves   ******************')

    fname = 'results/' + case + '_photoids_plasticc.dat'
    
    meta = pd.read_csv(fname, index_col=False)
    codes = np.unique(meta['code'].values)

    res = get_distances(fname,
                                   data_prefix='LSST_DDF',
                                   data_folder='/media/RESSPECT/data/PLAsTiCC/SNANA',            
                                   select_modelnum=None,
                                   salt2mu_prefix='test_salt2mu_res',
                                   maxsnnum=50000,
                                   select_orig_sample=['test'],
                                   salt3_outfile='salt3pipeinput.txt',
                                   data_prefix_has_sntype=False,
                                   master_fitres_name='results/master_fitres.fitres', 
                                   append_master_fitres=True,
                                   restart_master_fitres=restart_master)
    
    res['distances'].to_csv('results/mu_photoIa_plasticc_' + case + '.dat', index=False)
    res['cosmopars'].to_csv('results/cosmo_photoIa_plasticc_' + case + '.dat', index=False)
    
        
    print("--- %s seconds ---" % (time.time() - start_time))


# SALT2mu input file name
salt2mu_fname = 'SALT2mu.input'


if replace_z:
    if add_lowz:
        if bias:
            # path to lowz fitres
            fitres_lowz_fname = '/media/RESSPECT/data/temp_lowz_sim/lowz_only_fittres.fitres'
        
        else:
            raise ValueError('Low-z without bias not implemented yet.')
            
        fitres_lowz = pd.read_csv(fitres_lowz_fname, index_col=False, comment="#", 
                                  skip_blank_lines=True, delim_whitespace=True)
        
        fitres_lowz['zHD'] = fitres_lowz['SIM_ZCMB']

    # path to main fitres
    fitres_main_fname = 'results/master_fitres.fitres'
    
    # read fitres
    fitres_main = pd.read_csv(fitres_main_fname, index_col=False, comment="#", 
                          skip_blank_lines=True, delim_whitespace=True)

    if add_lowz:
        # join samples considering only common columns
        frames = [fitres_lowz, fitres_main]
        fitres = pd.concat(frames, ignore_index=True)
    else:
        fitres = fitres_main   
    
    # update redshift value
    fitres['zHD'] = fitres['SIM_ZCMB']

    # replace nans with number so SNANA recognizes the columns
    fitres.fillna(value=-99, inplace=True)

    # save combined fitres to file
    if add_lowz:
        if bias:
            fitres.to_csv('results/master_fitres_new_lowz_withbias.fitres', sep=" ", index=False)
        else:
            fitres.to_csv('results/master_fitres_new_lowz_nobias.fitres', sep=" ", index=False)
    else:
        fitres.to_csv('results/master_fitres_new.fitres', sep=" ", index=False)
    
samples_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/posteriors/' + case + '/'
if not os.path.isdir(samples_dir):
    os.makedirs(samples_dir)

# change parameters for SALT2mu
op = open(salt2mu_fname, 'r')
lin = op.readlines()
op.close()

lin[0] = 'bins=' + str(nbins) + '\n'


if add_lowz:
    if bias:
        lin[-3] = 'prefix=results/test_salt2mu_lowz_withbias_' + case + '\n'
        lin[-4] = 'file=results/master_fitres_new_lowz_withbias.fitres' + '\n'
        fitres_comb_fname = 'results/test_salt2mu_lowz_withbias_' + case + '.fitres'
        stan_input_fname = 'results/stan_input_salt2mu_lowz_withbias_' + case + '.csv'
    else:
        lin[-3] = 'prefix=results/test_salt2mu_lowz_nobias_' + case + '\n'
        lin[-4] = 'file=results/master_fitres_new_lowz_nobias.fitres' + '\n'
        fitres_comb_fname = 'results/test_salt2mu_lowz_nobias_' + case + '.fitres'
        stan_input_fname = 'results/stan_input_salt2mu_lowz_npbias_' + case + '.csv'
else:
    lin[-3] = 'prefix=results/test_salt2mu_' + case + '\n'
    lin[-4] = 'file=results/master_fitres_new.fitres' + '\n'
    fitres_comb_fname = 'results/test_salt2mu_' + case + '.fitres'
    stan_input_fname = 'results/stan_input_salt2mu_' + case + '.csv'

op2 = open(salt2mu_fname, 'w')
for line in lin:
    op2.write(line)
op2.close()

# get distances from SALT2MU
os.system('SALT2mu.exe ' + salt2mu_fname)

# read data for Bayesian model
fitres_comb = pd.read_csv(fitres_comb_fname, index_col=False, comment="#", skip_blank_lines=True, 
                           delim_whitespace=True)

# set initial conditions
z0 = 0
E0 = 0
c = 3e5
H0 = 70

# remove duplicated redshift
fitres_final = fitres_comb.drop_duplicates(subset=['SIM_ZCMB'], keep='first')

# order data according to redshift 
indx = np.argsort(fitres_final['SIM_ZCMB'].values)

# create input data
stan_input = {}
stan_input['nobs'] = fitres_final.shape[0]
stan_input['z'] = fitres_final['SIM_ZCMB'].values[indx]
stan_input['mu'] = fitres_final['MU'].values[indx]
stan_input['muerr'] = fitres_final['MUERR'].values[indx]
stan_input['z0'] = z0
stan_input['H0'] = H0
stan_input['c'] = c
stan_input['E0'] = np.array([E0])

# save only stan input to file
stan_input2 = {}
stan_input2['z'] = stan_input['z']
stan_input2['mu'] = stan_input['mu']
stan_input2['muerr'] = stan_input['muerr']

stan_input_tofile = pd.DataFrame(stan_input2)

stan_input_tofile[['z', 'mu', 'muerr']].to_csv(stan_input_fname, index=False)

stan_model="""
functions {
     /** 
     * ODE for the inverse Hubble parameter. 
     * System State E is 1 dimensional.  
     * The system has 2 parameters theta = (om, w)
     * 
     * where 
     * 
     *   om:       dark matter energy density 
     *   w:        dark energy equation of state parameter
     *
     * The system redshift derivative is 
     * 
     * d.E[1] / d.z  =  
     *  1.0/sqrt(om * pow(1+z,3) + (1-om) * (1+z)^(3 * (1+w)))
     * 
     * @param z redshift at which derivatives are evaluated. 
     * @param E system state at which derivatives are evaluated. 
     * @param params parameters for system. 
     * @param x_r real constants for system (empty). 
     * @param x_i integer constants for system (empty). 
     */ 
     real[] Ez(real z,
               real[] H,
               real[] params,
               real[] x_r,
               int[] x_i) {
           real dEdz[1];
           dEdz[1] = 1.0/sqrt(params[1]*(1+z)^3
                     +(1-params[1])*(1+z)^(3*(1+params[2])));
           return dEdz;
    } 
}
data {
    int<lower=1> nobs;              // number of data points
    real E0[1];                     // integral(1/H) at z=0                           
    real z0;                        // initial redshift, 0
    real c;                         // speed of light
    real H0;                        // hubble parameter
    real mu[nobs];                  // distance modulus
    vector[nobs] muerr;      // error in distance modulus
    real<lower=0> z[nobs];          // redshift
}
transformed data {
      real x_r[0];                  // required by ODE (empty)
      int x_i[0]; 
}
parameters{
      real<lower=0, upper=1> om;    // dark matter energy density
      real<lower=-2, upper=0> w;    // dark energy equation of state parameter
}
transformed parameters{
      real DC[nobs,1];                        // co-moving distance 
      real pars[2];                           // ODE input = (om, w)
      real dl[nobs];                          // luminosity distance
      real DH;                                // Hubble distance = c/H0
 
 
      DH = (c/H0);
      pars[1] = om;
      pars[2] = w;
      
      // Integral of 1/E(z) 
      DC = integrate_ode_rk45(Ez, E0, z0, z, pars,  x_r, x_i);
      for (i in 1:nobs) {
            dl[i] = 25 + 5 * log10(DH * (1 + z[i]) * DC[i, 1]);
      }
}
model{
      // priors and likelihood
      om ~ normal(0.3, 0.1);
      w ~ normal(-1, 0.2);

      mu ~ normal(dl, muerr);
}
generated quantities {
    vector[nobs] log_lik;
    vector[nobs] mu_hat;
    
    for (j in 1:nobs) {
        log_lik[j] = normal_lpdf(mu[j] | dl[j], muerr[j]);
        mu_hat[j] = normal_rng(dl[j], muerr[j]);
    }
}
"""

model = pystan.StanModel(model_code=stan_model)

fit = model.sampling(data=stan_input, iter=16000, chains=3, warmup=10000, control={'adapt_delta':0.99})

# print summary
res = fit.stansummary(pars=["om", "w"])
check = str(pystan.check_hmc_diagnostics(fit))
print(res)
print( ' ******* ')
print(check)


if add_lowz and bias:
    summ_fname = samples_dir + 'stan_summary_' + case + '_lowz_withbias.dat'
    summ_fname2 = 'results/stan_summary_' + case + '_lowz_withbias.dat'
    chains_fname = samples_dir + '/chains_' + case + '_lowz_withbias.pkl'
    trace_fname = samples_dir + '/trace_plot_' + case + '_lowz_withbias.png'
    trace_fname2 = 'results/trace_plot_' + case + '_lowz_withbias.png'
elif add_lowz and not bias:
    summ_fname = samples_dir + 'stan_summary_' + case + '_lowz_nobias.dat'
    summ_fname2 = 'results/stan_summary_' + case + '_lowz_nobias.dat'
    chains_fname = samples_dir + '/chains_' + case + '_lowz_nobias.pkl'
    trace_fname = samples_dir + '/trace_plot_' + case + '_lowz_nobias.png'
    trace_fname2 = 'results/trace_plot_' + case + '_lowz_nobias.png'
else:
    summ_fname = samples_dir + 'stan_summary_' + case + '.dat'
    summ_fname2 = 'results/stan_summary_' + case + '.dat'
    chains_fname = samples_dir + '/chains_' + case + '.pkl'
    trace_fname = samples_dir + '/trace_plot_' + case + '.png'
    trace_fname2 = 'results/trace_plot_' + case + '.png'

op2 = open(summ_fname, 'w')
op2.write(res)
op2.write('\n ************* \n')
op2.write(check)
op2.close()

samples = fit.extract(permuted=True)

pickle.dump(samples, open(chains_fname, "wb"))

pystan.check_hmc_diagnostics(fit)

# plot chains
import arviz
import matplotlib.pyplot as plt

arviz.plot_trace(fit, ['om', 'w'])
plt.savefig(trace_fname)

copyfile(trace_fname, trace_fname2)
copyfile(summ_fname, summ_fname2)