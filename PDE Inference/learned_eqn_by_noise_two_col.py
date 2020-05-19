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

#### This file creates tables 2 and 5 in our manuscript

dim = 1
algo_name = 'Greedy'
model = 'fkpp'
comp_str = "__ann" 
deg = 2

noise_list = ['01']#,'05'
pts_list = ['03','05','10']

sim_no_list = [1,3,2,4]
sim_str_list = ['Slow','Diffuse','Fast','Nodular']    

font = {'size': 18}
plt.rc('font', **font)


if dim == 1:
    
    true_derivs = ['u_{xx}','u','u^2']
    
    params_estimates = [[.03,3,-3],[.3,3,-3],[.3,30,-30],[.03,30,-30]]
    
elif dim == 2:
    
    true_derivs = ['u_{xx}','u_{yy}','u','u^2']

    params_estimates = [[.3,.3,30,-30],
                        [.03,.03,30,-30]]
        


for ni, n in enumerate(noise_list):

    print('\n')
    print('\\begin{table}')
    print('\\centering')
    print('\\begin{tabular}{|c|c|c|c|}')
    print('\\hline')

    for dind,sim_no in enumerate(sim_no_list):
        print('\\multicolumn{2}{|c|}{\\textbf{' +sim_str_list[dind]+' Population}} & \\multicolumn{2}{c}{$\\boldsymbol{'+print_pde_table(params_estimates[dind],true_derivs)+'}$} \\\\')
        print('\\hline' )
        print('$\\sigma$ & \\ $N$ & Learned Equation (0-0.5 yrs) & Learned Equation (0-3 yrs) \\\\ ')
        print('\\hline' )
        
        
        for j,pts in enumerate(pts_list):  

            print_str =  '     '+n+' & '+pts

            for time_str in ['_early_time','']:  

                filename = "pickle_data/"+algo_name+'_'+model+'_'+str(dim)+"d"+time_str+"_"+str(sim_no)+"_"+pts+"pts_"+n+comp_str+'_deg_'+str(deg) + '_xi_results_final'

                mat2 = np.load(filename+'.npy',allow_pickle=True, encoding='latin1').item()
                #mat2 = np.load(filename+'_'+method+'_results.npy').item()
                description = mat2['description']
                xi_final = mat2['final_xi']


                rows_to_keep   = np.where(xi_final != 0)[0]
                xi_tilde = xi_final[rows_to_keep]
                description_tilde = []
                for j in rows_to_keep:
                    description_tilde.append(description[j])

                if len(description_tilde) > 1:
                    if 'uu_{xx}' in description_tilde:
                        if 'u_{x}^2' not in description_tilde:
                            xi_tilde = np.delete(xi_tilde,description_tilde.index('uu_{xx}'))
                            description_tilde.remove('uu_{xx}')
                    if 'u_{x}^2' in description_tilde:
                        if 'uu_{xx}' not in description_tilde:
                            xi_tilde = np.delete(xi_tilde,description_tilde.index('u_{x}^2'))
                            description_tilde.remove('u_{x}^2')
                if len(description_tilde) > 1:
                    if 'u^2u_{xx}' in description_tilde:
                        if 'uu_{x}^2' not in description_tilde:
                            xi_tilde = np.delete(xi_tilde,description_tilde.index('u^2u_{xx}'))
                            description_tilde.remove('u^2u_{xx}')
                    if 'uu_{x}^2' in description_tilde:
                        if 'u^2u_{xx}' not in description_tilde:
                            xi_tilde = np.delete(xi_tilde,description_tilde.index('uu_{x}^2'))
                            description_tilde.remove('uu_{x}^2')
                
                print_str+= ' & $' + print_pde_table(xi_tilde,description_tilde,n=4)+'$'
            
            print_str += ' \\\\'
            print(print_str)
            print('\\hline')
        
        if dind < len(sim_no_list) - 1:
            print('\\multicolumn{4}{|c|}{ } \\\\')
            print('\\hline')

    print('\\end{tabular}')
    if time_str == '_early_time':
        print('\\caption{'+str(dim)+'d Results for all populations using subagging with '+n+'\% noisy  data on a short time interval (0-0.5 years).}')
    else:
        print('\\caption{'+str(dim)+'d Results for all populations using subagging with '+n+'\% noisy  data on a long time interval (0-3 years).}')
    print('\\end{table}')
    
     