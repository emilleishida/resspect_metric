import pandas as pd
import numpy as np
import pystan
import os
import pickle
import arviz
import matplotlib.pyplot as plt

from shutil import move


def create_directories(dir_output: str):
    """Create directory structure. 
    
    Parameters
    ----------
    dir_output: str
        Full path to main output directory. 
    """
    
    dirname_list = [dir_output ,
                dir_output + 'fitres/',
                dir_output + 'KLD/',
                dir_output + 'M0DIF/',
                dir_output + 'posteriors/',
                dir_output + 'posteriors/trace/',
                dir_output + 'posteriors/pkl/',
                dir_output + 'posteriors/csv/',
                dir_output + 'samples/',
                dir_output + 'COV/',
                dir_output + 'stan_input/',
                dir_output + 'stan_summary/',
                dir_output + 'Wasserstein/',
                dir_output + 'SALT2mu_input/']


    for fname in dirname_list:
        if not os.path.isdir(fname):
            os.makedirs(fname)


def read_fitres(fname_zenodo_meta: str,
                fname_sample: str,
                fname_fitres_lowz: str,
                fname_output: str,
                dir_output: str,
                sample: str, 
                lowz = True,
                to_file=False):
    """
    Read fitres file for a given sample.
    
    Parameters
    ----------
    fname_zenodo_meta: str
        Path to zenodo metadata file.
    fname_sample: str
        Path to sample file. 
    fname_fitres_lowz: str
        Path to lowz fitres file.
    fname_output: str
        Path to output file containing concatenated fitres.
    dir_output: str
        Output folder.
    sample: str
        Sample identifier. 
    lowz: bool (optional)
        If True add lowz sample. Default is True.
    to_file: bool (optional)
        If True, save concatenated fitres to file.
        Default is False.
        
    Returns
    -------
    pd.DataFrame
        Complete fitres data frame.
    """
    
    # read plasticc test metadata
    test_metadata = pd.read_csv(fname_test_zenodo_meta)

    # read sample 
    fitres_main = pd.read_csv(fname_sample, index_col=False)
    if ' ' in fitres_main.keys()[0]:
        fitres_main = pd.read_csv(fname_sample, delim_whitespace=True)

    # read lowz sample
    if lowz:
        fitres_lowz = pd.read_csv(fname_fitres_lowz, index_col=False, comment="#", 
                                  skip_blank_lines=True, delim_whitespace=True)
        fitres_lowz['zHD'] = fitres_lowz['SIM_ZCMB']
        fitres_lowz.fillna(value=-99, inplace=True)
        
        flag1 = np.array([item in fitres_main.keys() for item in fitres_lowz.keys()])
        flag2 = np.array([item in fitres_lowz.keys() for item in fitres_main.keys()])
        fitres = pd.concat([fitres_lowz[list(fitres_lowz.keys()[flag1])], 
                            fitres_main[list(fitres_main.keys()[flag2])]], 
                            ignore_index=True)
        
    else:
        fitres = fitres_main

    if to_file:
        fitres.to_csv(dir_output + 'fitres/master_fitres_' + sample + '_lowz_withbias.fitres', 
                      sep=" ", index=False)
        

def fit_salt2mu(biascorr_dir: str, sample: str, root_dir: str, 
                fname_input_salt2mu: str, fname_output_salt2mu:str, 
                nbins=30, field='DDF', biascorr=False):
    """
    Parameters
    ----------
    biascorr_dir: str
        Path to bias correction directory.
    sample: str
        Sample identification.
    root_dir: str
        Path to directory storing SALT2 fitres subfolder.
    fname_input_salt2mu: str
        Path to example input SALT2mu file.
    fname_output_salt2mu: str
        Path to output SALT2mu input file.
    nbins: int (optional)
        Number of bins for SALT2mu fit. Default is 30.
    field: str (optional)
        Observation strategy. DDF or WFD. Default is DDF.
    biascorr: bool (optional)
        If true add bias correction. Default is False.    
    """
    
    # change parameters for SALT2mu
    op = open(fname_input_salt2mu, 'r')
    lin = op.readlines()
    op.close()

    for i in range(len(lin)):
        if lin[i][:4] == 'bins':
            lin[i] = 'bins=' + str(nbins) + '\n'
        elif lin[i][:6] == 'prefix':
            lin[i] = 'prefix=test_salt2mu_lowz_withbias_' + sample + '\n'
        elif lin[i][:4] == 'file':
            lin[i] = 'file=' + root_dir + 'fitres/master_fitres_' + sample + \
                     '_lowz_withbias.fitres' + '\n'
        elif lin[i][:7] == 'simfile' and biascorr:
            lin[i] = 'simfile_biascor=' + biascorr_dir + '/LSST_' + field + \
           '_BIASCOR.FITRES.gz,' + biascorr_dir + '/FOUNDATION_BIASCOR.FITRES.gz\n'
        elif lin[i][:7] == 'simfile' and not biascorr:
            lin[i] = '\n'
        elif lin[i][:3] == 'opt' and not biascorr:
            lin[i] = '\n'

    op2 = open(fname_output_salt2mu, 'w')
    for line in lin:
        op2.write(line)
    op2.close()

    # get distances from SALT2MU
    os.system('SALT2mu.exe ' + fname_output_salt2mu)
  
    # put files in correct directory
    move('test_salt2mu_lowz_withbias_' + sample + '.COV', 
         root_dir + 'COV/test_salt2mu_lowz_withbias_' + sample + '.COV')
    move('test_salt2mu_lowz_withbias_' + sample + '.fitres', 
         root_dir + 'fitres/test_salt2mu_lowz_withbias_' + sample + '.fitres')
    move('test_salt2mu_lowz_withbias_' + sample + '.M0DIF', 
         root_dir + 'M0DIF/test_salt2mu_lowz_withbias_' + sample + '.M0DIF')
    
    
def remove_duplicated_z(fitres_final: pd.DataFrame):
    """
    Add small offset to avoid duplicated redshift values.
    
    Parameters
    ----------
    fitres_final: pd.DataFrame
        Data frame containing output from SALT2 fit.
        
    Returns
    -------
    pd.DataFrame
        Fitres data frame with updated redshifts.
    """
    
    z_all = []
    for j in range(fitres_final.shape[0]):
        z = fitres_final.iloc[j]['SIM_ZCMB']
        z_new = z
        if z in z_all:
            while z_new in z_all:
                z_new = z + np.random.normal(loc=0, scale=0.001)
            
        fitres_final.at[j, 'SIM_ZCMB'] = z_new
        z_all.append(z_new)
        
    return fitres_final


def fit_stan(fname_fitres_comb: str, dir_output: str, sample: str,
             screen=False, lowz=True, bias=True, plot=False):
    """
    Fit Stan model for w.
    
    Parameters
    ----------
    fname_fitres_comb: str
        Complete path to fitres (with lowz, if requested).
    dir_output: str
        Complete path to output root folder.
    sample: str
        Sample to be fitted.
    screen: bool (optional)
        If True, print Stan results to screen. Default is False.     
    lowz: bool (optional)
        If True, add low-z sample. Default is True.
    bias: bool (optional)
        If True, add bias correction. Default is True. 
    plot: bool (optional)
        If True, generate chains plot. Default is False.
    """

    # read data for Bayesian model
    fitres_final = pd.read_csv(fname_fitres_comb, index_col=False, comment="#", 
                          skip_blank_lines=True, delim_whitespace=True)

    # set initial conditions
    z0 = 0
    E0 = 0
    c = 3e5
    H0 = 70

    # add small offset to duplicate redshifts
    fitres_final = remove_duplicated_z(fitres_final)

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
    fname_stan_input = dir_output + 'stan_input/stan_input_salt2mu_lowz_withbias_' + sample + '.csv'

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

    if screen:
        print(res)
        print( ' ******* ')
        print(check)

    if lowz and bias:
        summ_fname = dir_output + 'stan_summary/stan_summary_' + sample + '_lowz_withbias.dat'
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '_lowz_withbias.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '_lowz_withbias.png'
    elif lowz and not bias:
        summ_fname = dir_output + 'stan_summary/stan_summary_' + sample + '_lowz_nobias.dat'
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '_lowz_nobias.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '_lowz_nobias.png'
    else:
        summ_fname = dir_output + 'stan_summary/stan_summary_' + sample + '.dat'
        chains_fname = dir_output + 'posteriors/pkl/chains_' + sample + '.pkl'
        trace_fname = dir_output + 'posteriors/trace/trace_plot_' + sample + '.png'

    op2 = open(summ_fname, 'w')
    op2.write(res)
    op2.write('\n ************* \n')
    op2.write(check)
    op2.close()

    samples = fit.extract(pars=['om', 'w'], permuted=True, inc_warmup=False)

    pickle.dump(samples, open(chains_fname, "wb"))

    pystan.check_hmc_diagnostics(fit)

    ### plot chains
    if plot:
        arviz.plot_trace(fit, ['om', 'w'])
        plt.savefig(trace_fname)

    if lowz and bias:
        data = pd.read_pickle(chains_fname)
        data2 = pd.DataFrame(data)
        data2.to_csv(dir_output + 'posteriors/csv/' + \
                     'chains_'  + sample + '_lowz_withbias.csv.gz', index=False)




##################    user choices     ##########################################
#sample = '83SNIa17SNII'                        # choose sample case
nobjs = '3000'
nbins = 30                                    # number of bins for SALT2mu
om_pri = [0.3, 0.01]                          # gaussian prior on om => [mean, std]
w_pri = [-10, 9]                              # flat prior on w
lowz = True                                   # choose to add lowz sample
field = 'WFD'                                 # choose field
version = '2'                                 # realization or version
biascorr = True
screen = True

save_full_fitres = True
plot_chains = True

# path to input (auxiliary files) and output directories
if biascorr:
    output_root = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data3/' + field + '/results/'
else:
    output_root = field + '/results_nobias/'
    
dir_output = output_root + 'v' + version + '/' + str(nobjs) + '/'
#dir_input = '/media/RESSPECT/data/for_metrics/final_data/misc/'
dir_input = '/media2/RESSPECT2/clean_output/misc/'

# path to input files in the coin01 server
fname_test_zenodo_meta = '/media/RESSPECT/data/PLAsTiCC/PLAsTiCC_zenodo/plasticc_test_metadata.csv'

biascorr_dir = '/media/RESSPECT/data/PLAsTiCC/biascorsims/'

fname_input_salt2mu = 'template_SALT2mu.input'
fname_output_salt2mu = 'SALT2mu.input'

fname_fitres_lowz = dir_input + 'lowz_only_fittres.fitres'
samples_dir = '/media/RESSPECT/data/PLAsTiCC/for_metrics/final_data3/' + field + '/results/v' + str(version) + '/' + str(nobjs) + '/samples/'


###################################################################################

# read samples
samples_list = os.listdir(samples_dir)

#done = ['random3000.csv', 'fiducial3000.csv', '99SNIa1SNIax.csv', 
#        '98SNIa2SNIax.csv', '95SNIa5SNIax.csv', '90SNIa10SNIax.csv',  '99.9SNIa0.1SNIax.csv']
done = ['random' + nobjs +'.csv', 'fiducial' + nobjs +'.csv', 'perfect' + nobjs + '.csv']

for name in samples_list:
    if name in done:
        sample = name[:-4]
        
        fname_sample = samples_dir + sample +'.csv'
        fname_fitres_comb = dir_output + 'fitres/test_salt2mu_lowz_withbias_' + sample + '.fitres'
        
        ##############################
        ### Create directories #######
        ##############################

        create_directories(dir_output=dir_output)
        
        #######################
        ### Generate fitres ###
        #######################

        read_fitres(fname_zenodo_meta=fname_test_zenodo_meta,
                    fname_sample=fname_sample,
                    fname_fitres_lowz=fname_fitres_lowz,
                    fname_output=fname_output_salt2mu,
                    dir_output=dir_output,
                    sample=sample, 
                    lowz = lowz,
                    to_file=save_full_fitres)



        #######################
        ####### SALT2mu #######
        #######################

        fit_salt2mu(biascorr_dir=biascorr_dir, 
            sample=sample, root_dir=dir_output, 
            fname_input_salt2mu=fname_input_salt2mu, fname_output_salt2mu=fname_output_salt2mu, 
            nbins=nbins, field=field, biascorr=biascorr)

       #######################
       #######   wfit  #######
       #######################

        # get current location
        dir_work = os.getcwd()

        # change working directory
        os.chdir(dir_output + 'M0DIF/')

        # run wfit
        os.system('wfit.exe test_salt2mu_lowz_withbias_' + sample + '.M0DIF -ompri ' + \
          str(om_pri[0]) + ' -dompri ' + str(om_pri[1]) + \
          ' -hmin 70 -hmax 70 -hsteps 1 -wmin -10 -wmax 9 -ommin ' + \
          str(om_pri[0] - om_pri[1]) + ' -ommax ' + str(om_pri[0] + om_pri[1]))

        # go back to working directory
        os.chdir(dir_work)
        
        # move cospar files to appropriate directory
        move(dir_output + 'M0DIF/test_salt2mu_lowz_withbias_' + sample + '.M0DIF.cospar', 
             dir_output + 'cospar/test_salt2mu_lowz_withbias_' + sample + '.M0DIF.cospar')

        #######################
        ##### Stan model ######
        #######################
        
        fit_stan(fname_fitres_comb=fname_fitres_comb, dir_output=dir_output, sample=sample,
                 screen=screen, lowz=lowz, bias=biascorr, plot=plot_chains)