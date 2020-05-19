import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from PDE_FIND2 import *
from matplotlib.patches import Polygon
import os
import scipy.io as sio
import scipy.optimize
import itertools
import pickle
import statistics
import seaborn as sns
import pandas as pd
import glob
import statsmodels.stats.stattools

import pdb


#This code creates Tables 3 and 4 in our manuscript

dim = 1
algo_name = 'Greedy'
model = 'fkpp'
comp_str = "__ann" 
deg = 2

noise_list = ['05']
pts_list = ['05']

sim_no_list = [1,2,3,4]
sim_str_list = ['Slow','Fast','Diffuse','Nodular']    

time_str = '' #'_early_time'

font = {'size': 18}
plt.rc('font', **font)


if dim == 1:
    
    true_derivs = ['u_{xx}','u','u^2']
    
    params_estimates = [[3,3],[30,30],[30,3],[3,30]]
    
elif dim == 2:
    
    true_derivs = ['u_{xx}','u_{yy}','u','u^2']

    params_estimates = [[.3,.3,30,-30],
                        [.03,.03,30,-30]]
        


for ni, n in enumerate(noise_list):

    print('\n')
    print('\\begin{table}')
    print('\\centering')
    print('\\begin{tabular}{|c|c|c|c|c|c|}')
    print('\\hline')
    print(' & Parameter & True & Median & \% Error & Normalized SE \\\\ ')
    print('\\hline')

    for dind,sim_no in enumerate(sim_no_list):
        
        for j,pts in enumerate(pts_list):    
            
            filename = "pickle_data/"+algo_name+'_'+model+'_'+str(dim)+"d"+time_str+"_"+str(sim_no)+"_"+pts+"pts_"+n+comp_str+'_deg_'+str(deg) + '_param_est_results'
            
            mat = np.load(filename+'.npy', allow_pickle=True, encoding='latin1').item()
            xi_list = mat['xi_list']
            description = mat['description']

            xi = np.squeeze(np.array(xi_list))
            xi_median = np.median(xi_list,axis=0)
            xi_std = np.std(xi_list,axis=0)

            D_true = params_estimates[sim_no-1][0]
            D_est = xi_median[description.index('u_{xx}')][0]*100
            D_std = xi_std[description.index('u_{xx}')][0]*100
            D_rel_std = D_std/D_est
            D_rel_err = np.abs((D_est-D_true)/D_true)


            r_true = params_estimates[sim_no-1][1]
            r_est = xi_median[description.index('u(1-u)')][0]
            r_std = xi_std[description.index('u(1-u)')][0]
            r_rel_std = r_std/r_est
            r_rel_err = np.abs((r_est-r_true)/r_true)

            print('\\multirow{2}*{\\textbf{'+sim_str_list[dind]+' Population}} & $D$ & ' +str(D_true)+ ' & ' + str(np.round(D_est,3)) + ' & ' + str(np.round(100*D_rel_err,1)) + '\% & '  + str(np.round(D_rel_std,3)) + '\\\\')
            print(' & $r$ & ' +str(r_true)+ ' & ' + str(np.round(r_est,3)) +'  & '+ str(np.round(r_rel_err*100,1)) + '\% & ' + str(np.round(r_rel_std,3)) + '\\\\')
            print('\\hline')

            

    print('\\end{tabular}')
    if time_str == '_early_time':
        print('\\caption{'+str(dim)+'d parameter estimation and uncertainty quantification results for all populations using subagging with '+n+'\% noisy  data on a short time interval (0-0.5 years). \\label{tab:UQ_short}}')
    else:
        print('\\caption{'+str(dim)+'d parameter estimation and uncertainty quantification results for all populations using subagging with '+n+'\% noisy  data on a long time interval (0-3 years). \\label{tab:UQ_long}}')
    print('\\end{table}')
     