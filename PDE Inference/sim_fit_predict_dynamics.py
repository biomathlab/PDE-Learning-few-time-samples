import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from PDE_FIND2 import *
from model_selection_IP import learned_RHS, PDE_sim
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
from scipy import interpolate
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf') 
from scipy.signal import savgol_filter

import pdb

#This code is used to create figures 5, 8-11 in our manuscript.

dim = 1
algo_name = 'Greedy'
model = 'fkpp'
comp_str = "__ann"
deg = 2

method = '' 

noise_list = ['01']#,'05'


font = {'size'   : 22}

plt.rc('font', **font)

if dim == 1:
    
	true_derivs = ['u_{xx}','u','u^2']
	sim_no_list = [1,3,2,4]
	sim_str_list = ['Slow','Diffuse','Fast','Nodular']    

	params_estimates = [[.03,3,-3],[.3,30,-30],[.3,3,-3],[.03,30,-30]]
	pts_list = ['05']#['03','05','10']

   


for time_ind,time_str in enumerate(['_early_time','']):
	for dind,sim_no in enumerate(sim_no_list):

		for ni, n in enumerate(noise_list):
	        
			for j,pts in enumerate(pts_list):    

				#####
				#load in inferred EQN
				#####

				filename = "pickle_data/"+algo_name+'_'+model+'_'+str(dim)+"d"+time_str+"_"+str(sim_no)+"_"+pts+"pts_"+n+comp_str+'_deg_'+str(deg)
				
				mat2 = np.load(filename+'_xi_results_final.npy', allow_pickle=True, encoding='latin1').item()
				xi_final = mat2['final_xi']
				
				description = mat2['description']
				


				rows_to_keep   = np.where(xi_final != 0)[0]
				xi_tilde = xi_final[rows_to_keep]
				description_tilde = []
				for j in rows_to_keep:
				    description_tilde.append(description[j])
				if 'uu_{xx}' in description_tilde:
				    if 'u_{x}^2' not in description_tilde:
				        xi_tilde = np.delete(xi_tilde,description_tilde.index('uu_{xx}'))
				        description_tilde.remove('uu_{xx}')
				if 'u^2u_{xx}' in description_tilde:
				    if 'uu_{x}^2' not in description_tilde:
				        xi_tilde = np.delete(xi_tilde,description_tilde.index('u^2u_{xx}'))
				        description_tilde.remove('u^2u_{xx}')
				if 'u_{x}^2' in description_tilde:
				    if 'uu_{xx}' not in description_tilde:
				        xi_tilde = np.delete(xi_tilde,description_tilde.index('u_{x}^2'))
				        description_tilde.remove('u_{x}^2')
				if 'uu_{x}^2' in description_tilde:
				    if 'u^2u_{xx}' not in description_tilde:
				        xi_tilde = np.delete(xi_tilde,description_tilde.index('uu_{x}^2'))
				        description_tilde.remove('uu_{x}^2')

				#final equation form
				xi_final_full = xi_convert_full(xi_tilde,description_tilde,description)

				######
				#load in clean data
				######

				mat = np.load('data/fisher_KPP_'+str(dim)+'D_'+str(sim_no)+'_00.npy', allow_pickle=True, encoding='latin1').item()

				U_clean = mat['U'] # (199, 99)
				t_clean = mat['t']
				x_clean = mat['x']

				t0 = 5
				if time_str == '':
				    tf = -1
				elif time_str == '_early_time':
				    tf = 19

				if tf == -1:
					step_size = int((len(t_clean)-t0)/5)
				else:
					step_size = int((tf-t0)/5)

				U_clean = U_clean[:,t0:tf:step_size][:,:5]
				t_clean = t_clean[t0:tf:step_size][:5]
				
				mat_sf = np.load('surface_data/fkpp_' +str(dim)+"d"+time_str+"_"+str(sim_no)+"_"+pts+"pts_"+n+comp_str[1:]+ '.npy',allow_pickle=True, encoding='latin1').item()
				var_names = mat_sf['variable_names']

				if dim == 1:
					#variables
					X = mat_sf['variables'][0]
					T = mat_sf['variables'][1]
					
					U_sf = mat_sf['variables'][var_names.index('u')]

				x = np.unique(X)
				t = np.unique(T)
				
				''' Playing with simulating the smoothened data '''
				mat_data = np.load('data/fkpp_' +str(dim)+"d"+time_str+"_"+str(sim_no)+"_"+pts+"pts_"+n+ '.npy',allow_pickle=True, encoding='latin1').item()
				U_data = mat_data['outputs'][T==t[0]][:,0]
				
				x_plot = np.linspace(x[0],x[-1],200)
				IC = savgol_filter(U_data,5,3)
				IC[IC<0] = 0
				
				f_interpolate = interpolate.interp1d(x,IC)
				IC = f_interpolate(x_plot)


				RHS = learned_RHS  

				y_tmp = PDE_sim(xi_final_full,RHS,x_plot,t_clean,IC,description=description)

				fig = plt.figure(figsize=(10,8))
				ax = fig.add_subplot(1,1,1)
				colors = 'bgrkm'
				for i in np.arange(5):ax.plot(x_clean,U_clean[:,i],colors[i]+'.',markersize=15)
				for i in np.arange(5):ax.plot(x_plot,y_tmp[:,i],colors[i]+'-',label="$t$ = " + str(round(t_clean[i],2)))
				plt.xlim((x[0],x[-1]))
				plt.xlabel('$x$ (cm)')
				plt.ylabel('$u(t,x)$ (normalized)')
				if time_str == '':
					plt.title(sim_str_list[dind] + " Population, N = " + pts + ", \n fit dynamics for [0,3]")
				if time_str == '_early_time':
					plt.title(sim_str_list[dind] + " Population, N = " + pts + ", \n fit dynamics for [0,0.5]")
				plt.legend()
				
				if time_ind == 0: lht = "a"
				if time_ind == 1: lht = "c"
				ax.text(-.1, 1.1,lht + ')',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontsize=22)
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				plt.savefig("figures/dynamics_sim_"+str(sim_no-1)+"_"+time_str+'_'+pts + "_noise_" + n +".pdf", format='pdf')

				#########################
				####### Now attempt to Predicted dynamics
				#########################

				U_clean = mat['U'] 
				t_clean = mat['t']
				x_clean = mat['x']

				t0 = 5
				if time_str == '':
				    tf = 19
				    time_str_predict = '_early_time'
				elif time_str == '_early_time':
				    tf = -1
				    time_str_predict = ''

				if tf == -1:
					step_size = int((len(t_clean)-t0)/5)
				else:
					step_size = int((tf-t0)/5)

				U_clean = U_clean[:,t0:tf:step_size][:,:5]
				t_clean = t_clean[t0:tf:step_size][:5]
		


				mat_sf = np.load('surface_data/fkpp_' +str(dim)+"d"+time_str_predict+"_"+str(sim_no)+"_"+pts+"pts_"+n+comp_str[1:]+ '.npy',allow_pickle=True, encoding='latin1').item()
				var_names = mat_sf['variable_names']

				if dim == 1:
					#variables
					#X = mat['inputs'][:,0]
					#T = mat['inputs'][:,1]
					X = mat_sf['variables'][0]
					T = mat_sf['variables'][1]
					#U = mat['outputs'][:,0]
					U_sf = mat_sf['variables'][var_names.index('u')]

				x = np.unique(X)
				t = np.unique(T)


				''' Playing with simulating the smoothened data '''
				mat_data = np.load('data/fkpp_' +str(dim)+"d"+time_str_predict+"_"+str(sim_no)+"_"+pts+"pts_"+n+ '.npy',allow_pickle=True, encoding='latin1').item()
				U_data = mat_data['outputs'][T==t[0]][:,0]

				
				x_plot = np.linspace(x[0],x[-1],200)
				IC = savgol_filter(U_data,5,3)
				IC[IC<0] = 0
				
				f_interpolate = interpolate.interp1d(x,IC)
				IC = f_interpolate(x_plot)


				RHS = learned_RHS  

				y_tmp = PDE_sim(xi_final_full,RHS,x_plot,t_clean,IC,description=description)

				fig = plt.figure(figsize= (10,8))
				ax = fig.add_subplot(1,1,1)
				colors = 'bgrkm'
				for i in np.arange(5):ax.plot(x_clean,U_clean[:,i],colors[i]+'.',markersize=15)
				for i in np.arange(5):ax.plot(x_plot,y_tmp[:,i],colors[i]+'-',label="$t$ = " + str(round(t_clean[i],2)))
				plt.xlim((x[0],x[-1]))

				plt.xlabel('$x$ (cm)')
				plt.ylabel('$u(t,x)$ (normalized)')
				if time_str == '':
					plt.title(sim_str_list[dind] + " Population, N = " + pts + ", \n predicted dynamics for [0,0.5]")
				if time_str == '_early_time':
					plt.title(sim_str_list[dind] + " Population, N = " + pts + ", \n predicted dynamics for [0,3]")
				plt.legend()
				if time_ind == 0: lht = "b"
				if time_ind == 1: lht = "d"
				ax.text(-.1, 1.1,lht + ')',horizontalalignment='center',verticalalignment='center',transform = ax.transAxes,fontsize=22)
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

				plt.savefig("figures/dynamics_predict_"+str(sim_no-1)+"_"+time_str+'_'+pts + "_noise_" + n +".pdf", format='pdf')









