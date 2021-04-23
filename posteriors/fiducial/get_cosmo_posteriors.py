import pandas as pd
import numpy as np
import pystan
import os
import pickle

# number of bins for SALT2mu
nbins = 70

# rather to re-write fitres file
replace_z = True
add_lowz  = True

# SALT2mu input file name
salt2mu_fname = 'SALT2mu.input'

if replace_z:
    if add_lowz:
        # path to lowz fitres
        fitres_lowz_fname = '/media/RESSPECT/data/temp_lowz_sim/lowz_only_fittres.fitres'
        
        fitres_lowz = pd.read_csv(fitres_lowz_fname, index_col=False, comment="#", 
                              skip_blank_lines=True, delim_whitespace=True)
        
        fitres_lowz['zHD'] = fitres_lowz['SIM_ZCMB']

    # path to main fitres
    fitres_main_fname = '/media/emille/git/COIN/RESSPECT_work/PLAsTiCC/perfect_classifier/master_fitres_old.fitres'
    
    # read fitres
    fitres_main = pd.read_csv(fitres_main_fname, index_col=False, comment="#", 
                          skip_blank_lines=True, delim_whitespace=True)

    # update redshift value
    fitres_main['zHD'] = fitres_main['SIM_ZCMB']

    if add_lowz:
        # join samples considering only common columns
        frames = [fitres_lowz, fitres_main]
        fitres = pd.concat(frames, ignore_index=True)
    else:
        fitres = fitres_main   

    # replace nans with number so SNANA recognizes the columns
    fitres.fillna(value=-99, inplace=True)

    # save combined fitres to file
    fitres.to_csv('master_fitres_new.fitres', sep=" ", index=False)
    

if not os.path.isdir(str(nbins) + 'bins/'):
    os.makedirs(str(nbins) + 'bins/')

# change parameters for SALT2mu
op = open(salt2mu_fname, 'r')
lin = op.readlines()
op.close()

lin[0] = 'bins=' + str(nbins) + '\n'
lin[-3] = 'prefix=' + str(nbins) + 'bins/test_salt2mu_' + str(nbins) + 'bins\n'

op2 = open(salt2mu_fname, 'w')
for line in lin:
    op2.write(line)
op2.close()
    
# get distances from SALT2MU
os.system('SALT2mu.exe ' + salt2mu_fname)

# read data for Bayesian model
fitres_comb_fname = str(nbins) + 'bins/test_salt2mu_' + str(nbins) + 'bins.fitres'
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

stan_input_tofile[['z', 'mu', 'muerr']].to_csv( str(nbins) + 'bins/stan_input_perfect_classifier_salt2mu_' + str(nbins) + 'bins.csv', index=False)

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
      real Mint;                    // intrinsic magnitude
      real<lower=-2, upper=0> w;    // dark energy equation of state parameter
}
transformed parameters{
      real DC[nobs,1];                        // co-moving distance 
      real pars[2];                           // ODE input = (om, w)
      vector[nobs] mag;                       // apparent magnitude
      real dl[nobs];                          // luminosity distance
      real DH;                                // Hubble distance = c/H0
 
 
      DH = (c/H0);
      pars[1] = om;
      pars[2] = w;
      
      // Integral of 1/E(z) 
      DC = integrate_ode_rk45(Ez, E0, z0, z, pars,  x_r, x_i);
      for (i in 1:nobs) {
            dl[i] = 25 + 5 * log10(DH * (1 + z[i]) * DC[i, 1]/10);
            mag[i] = Mint + dl[i];
      }
}
model{
      // priors and likelihood
      om ~ normal(0.3, 0.1);
      Mint ~ normal(-19, 5); 
      w ~ normal(-1, 0.2);

      mu ~ normal(mag, muerr);
}
generated quantities {
    vector[nobs] log_lik;
    vector[nobs] mu_hat;
    
    for (j in 1:nobs) {
        log_lik[j] = normal_lpdf(mu[j] | mag[j], muerr[j]);
        mu_hat[j] = normal_rng(mag[j], muerr[j]);
    }
}
"""

model = pystan.StanModel(model_code=stan_model)

fit = model.sampling(data=stan_input, iter=6000, chains=3, warmup=3000)#, control={'max_treedepth':15})

# print summary
print(fit.stansummary(pars=["om", "w", "Mint"]))

samples = fit.extract()

pickle.dump(samples, open(str(nbins) + 'bins/chains.pkl', "wb"))

pystan.check_hmc_diagnostics(fit)

# plot chains
import arviz
import matplotlib.pyplot as plt

arviz.plot_trace(fit, ['om', 'w', 'Mint'])
plt.savefig(str(nbins) + 'bins/trace_plot_perfect_classifier_salt2mu_' + str(nbins) + 'bins.png')