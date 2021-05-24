import numpy as np
import pandas as pd
import pystan


# read Stan input
fname = 'stan_input_salt2mu_lowz_withbias_perfect3000.csv'
fitres_comb = pd.read_csv(fname)

# set initial conditions
z0 = 0
E0 = 0
c = 3e5
H0 = 70

# set prior on om
om_pri = [0.3, 0.05]

# add small perturbation to repeated redshfits
# this is done because for a few metrics we need to have the exact same
# number of data for all trials
z_all = []
for j in range(fitres_comb.shape[0]):
    z = fitres_comb.iloc[j]['z']
    z_new = z
    if z in z_all:
        while z_new in z_all:
            z_new = z + np.random.normal(loc=0, scale=0.001)
            
    fitres_comb.at[j, 'z'] = z_new
    z_all.append(z_new)

fitres_final = fitres_comb

# order data according to redshift 
indx = np.argsort(fitres_final['z'].values)

# create input data
stan_input = {}
stan_input['nobs'] = fitres_final.shape[0]
stan_input['z'] = fitres_final['z'].values[indx]
stan_input['mu'] = fitres_final['mu'].values[indx]
stan_input['muerr'] = fitres_final['muerr'].values[indx]
stan_input['z0'] = z0
stan_input['H0'] = H0
stan_input['c'] = c
stan_input['E0'] = np.array([E0])
stan_input['ompri'] = om_pri[0]
stan_input['dompri'] = om_pri[1]

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
    real ompri;
    real dompri;
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
      om ~ normal(ompri, dompri);
      w ~ normal(-1, 0.1);

      mu ~ normal(dl, muerr);
}
"""

model = pystan.StanModel(model_code=stan_model)

fit = model.sampling(data=stan_input, iter=10000, chains=5, warmup=9500, control={'adapt_delta':0.99})

# print summary
res = fit.stansummary(pars=["om", "w"])
check = str(pystan.check_hmc_diagnostics(fit))
print(res)
print( ' ******* ')
print(check)







