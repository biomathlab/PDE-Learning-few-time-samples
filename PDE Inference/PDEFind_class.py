import numpy as np
import pdb
import os
import time 
from collections import Counter
from tqdm import tqdm
from scipy.signal import savgol_filter
from PDE_FIND2 import *
from model_selection_IP import *

class PDE_Findclass(object):
    
    def __init__(self,
                 data_file,
                 comp_str,
                 trainPerc=0.5,
                 valPerc=0.5,
                 data_dir = "data/",
                 write_dir = "pickle_data/",
                 algo_name = "Greedy",
                 shuf_method = "bins",
                 prune_level = 0.05,
                 deg = 2,
                 reals = 100,
                 deriv_index = None,
                 print_pdes = False,
                 save_xi = False,
                 save_learned_xi = False,
                 num_eqns = 3,
                 save_learned_eqns=False,
                 true_param = None,
                 true_derivs = None,
                 xi_mod = ''):


        self.data_file = data_file
        #where data located
        self.data_dir = [data_dir + d  + comp_str for d in data_file]
        #where analytical data located
        self.true_data_dir = [data_dir + d for d in data_file]
        #where to write results
        self.write_dir = [write_dir + algo_name +'_' + d + '_' + comp_str 
                          + xi_mod + '_deg_' + str(deg) for d in data_file]
        
        #computation method
        self.comp_str = comp_str
        
        if not os.path.exists(write_dir):
            print("creating file " + write_dir)
            os.makedirs(write_dir)
    
        
        #for PDE-FIND implementation
        self.algo_name = algo_name
        self.trainPerc = trainPerc
        self.valPerc = valPerc
        self.prune_level = prune_level
        self.deg = deg
        self.reals = reals
        self.shuf_method = shuf_method
        if deriv_index == None:
            self.deriv_index = [None for d in self.data_dir]
        else:
            self.deriv_index = deriv_index
        self.print_pdes = print_pdes
        self.save_xi = save_xi
        self.save_learned_xi = save_learned_xi
        self.num_eqns = num_eqns
        self.save_learned_eqns = save_learned_eqns
        self.true_param = true_param
        self.true_derivs = true_derivs

    def train_val_PDEFind(self):
        
        '''
        Loads in surface-fit data & derivative files at specified location. 
        Constructs library of the potential right hand side terms, theta, and then performs
        model inference self.reals times. Each round of inference is followed by a round of
        pruning with pruning level of self.prune_level. All final inferred models are recorded and saved.

        Input:
            
        Output:
        
        '''


        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy', allow_pickle=True, encoding='latin1').item() for d in self.data_dir]

        X,T,Ut,Theta,description = theta_construct_1d(mat,self.deg)

        self.description = description
        x = np.unique(X)
        t = np.unique(T)
        
        print("Loaded in data for PDE-FIND, now determining top Models from "+str(self.reals)+" Simulations...")


        #now perform the training "reals" times
        for r in tqdm(np.arange(self.reals)):
            for i,d in enumerate(self.data_dir):
                
                
                #split data into training, validation data
                utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                
                #estimate xi
                xi, hparams, val_score,num_nonzero_u = run_PDE_Find_train_val(thetaTrain,utTrain,thetaVal,utVal,self.algo_name,self.description,lambda_lb=-3,lambda_ub=1,deriv_list=self.deriv_index[i])
                
                #record validation score
                val_score_0 = np.min(val_score)
                
                #perform pruning
                if len(xi[xi!=0]) > 1:
                    xi_new, val_score_new = PDE_FIND_prune_lstsq(xi,utTrain,utVal,thetaTrain,thetaVal,self.description,val_score_0,self.prune_level)
                    
                else:
                    #don't prune if xi only have one nonzero vector
                    xi_new = xi
                
                #append to list of xi values
                xi_list_no_prune[i].append(xi)
                xi_list[i].append(xi_new)


                #save results?
                if self.save_xi == True:
                    np.savez(self.write_dir[i],
                    xi_list = xi_list[i],
                    xi_list_no_prune=xi_list_no_prune[i],
                    description=self.description)


    def list_common_eqns(self):
        
        '''
        Loads in list of inferred model forms and determines the 3 most-frequently recovered forms. 
        Once these forms have been recovered, then we also estimate the parameters and standard 
        deviations of parameters for each model instance by finding their means and standard deviations
        from each instance where the model form was the final inferred model.

        Input:
            
        Output:
        
        '''

        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        xi_vote = [[] for d in self.data_dir]
        xi_vote_params = [[] for d in self.data_dir]
        xi_vote_params_SD = [[] for d in self.data_dir]

        for i,d in enumerate(self.data_dir):
            if os.path.isfile(self.write_dir[i]+".npz"):

                data = np.load(self.write_dir[i]+".npz",allow_pickle=True, encoding='latin1')
                xi_vote_tmp = []
                for j in range(len(data['xi_list'])):
                    xi_vote_tmp.append(trans_rev((data['xi_list'][j] != 0)*1)[0])
                xi_vote_tmp = Counter(xi_vote_tmp).most_common(self.num_eqns)
                xi_vote[i] = [x[0] for x in xi_vote_tmp]
                
                #help with bookkeeping for obtaining param estimates
                matrix_vote_initialized = [False for j in np.arange(self.num_eqns)]
                A = ["" for j in np.arange(self.num_eqns)]

                #loop through xi estimates
                for j in np.arange(len(data['xi_list'])):
                    xi_full = data['xi_list'][j]
                    #find if this xi estimate matches one of our top votes
                    match =  trans_rev(xi_full != 0)*1 == xi_vote[i]
                    #if so, add to list in the entry corresponding to that vote
                    if np.any(match):
                        if not matrix_vote_initialized[np.where(match)[0][0]]:
                            A[np.where(match)[0][0]] = xi_full
                            matrix_vote_initialized[np.where(match)[0][0]] = True
                        else:
                            A[np.where(match)[0][0]] = np.hstack((A[np.where(match)[0][0]],xi_full))

                #save params of mean parameter estimates for each equation
                if len(xi_vote_params[i]) == 0:
                    xi_vote_params[i] = [np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
                else:
                    xi_vote_params[i].append([np.mean(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])
                    
                #save SD of mean parameter estimates for each equation
                if len(xi_vote_params_SD[i]) == 0:
                    xi_vote_params_SD[i] = [np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))]
                else:
                    xi_vote_params_SD[i].append([np.std(A[k],axis=1) for k in np.arange(len(xi_vote[i]))])

            else:
                print("file "+self.write_dir[i]+".npz not found")
        
        #print most common equations
        if self.print_pdes:
            for i,d in enumerate(self.data_dir):
                #open file
                if i==0:
                    if self.save_learned_eqns:
                        text_f = open(self.write_dir[i]+"_learned_eqn.tex","w")
            
                if self.save_learned_eqns:
                    if i == 0:
                        text_f.write("\n\\begin{verbatim}")
                        text_f.write("\nFor simulation " + self.data_file[0] + ", PDEFIND with "+self.comp_str+" predicts the following equations:\n\\end{verbatim}\n\\begin{equation}\n\\begin{aligned}")
                for j in np.arange(np.min((self.num_eqns,len(xi_vote_params[i])))):
                    print(str(j+1)+ ". ")
                    write_pde(xi_vote_params[i][j],data['description'],ut=data_description[i]+'_t')
                        
            if self.save_learned_eqns:
                text_f.write("\n\\end{aligned}\n\\end{equation}")
                text_f.close()
                
        if self.save_learned_xi:
            for i,d in enumerate(self.data_dir):
                #save vectors of most common equations:
                data_xi = {}
                data_xi['xi_vectors'] = xi_vote_params[i]
                data_xi['xi_SD_vectors'] = xi_vote_params_SD[i]
                data_xi['description'] = data['description']
                data_xi['data_description']= data_description
                np.save(self.write_dir[i]+"_xi_results",data_xi)
        
    
    def simulate_learned_eqns_compare(self):


        '''
        Loads in noisy data & top inferred equation forms. 
        Each equation form will be simulated, starting from the noisy data at the first time point.
        Each model simulation will then be compared to the final data timepoint using AIC. Whichever
        model form has the lowest AIC score will be inferred as the final model form.
        Input:
            
        Output:
        
        '''

        #load in data source
        mat = [np.load('data/'+d+'.npy', allow_pickle=True, encoding='latin1').item() for d in self.data_file][0]
        
        #variables
        X = mat['inputs'][:,0]
        T = mat['inputs'][:,1]
        U = mat['outputs'][:,0]
        t = np.unique(T)

        #spatial grid of first, last time points
        t_ind_1 = 0
        t_ind_2 = len(t)-1

        X1 = X[T==t[t_ind_1]]

        #U values at first, last time points
        U1 = np.squeeze(U[T==t[t_ind_1]])
        U2 = U[T==t[t_ind_2]]

        #concatenate first, last timepoints:
        t_sim = np.array([t[t_ind_1],t[t_ind_2]])

        IC = np.zeros(U2.shape)
        IC = U1    
            
        #load in learned equations
        le_mat = np.load(self.write_dir[0]+"_xi_results.npy", allow_pickle=True, encoding='latin1').item()
        description = le_mat['description'].tolist()
        xi_list = le_mat['xi_vectors']
        
        #RHS for PDE simulation
        RHS = learned_RHS  

        #initialize u simulations, AIC score
        u_sim = []
        AIC = []
        N = len(U2)

        #simulate each learned model
        for i,xi in enumerate(xi_list):
            
            #simulate PDE
            y_tmp = PDE_sim(xi,RHS,X1,t_sim,IC,description=description)
            #recover, record final timepoint
            y_tmp = y_tmp[:,1:]
            u_sim.append(y_tmp)

            #record AIC score on final time point
            AIC.append(2*np.sum(xi!=0) + N*np.log(RSS_GLS(y_tmp[:,0],U2,0.5)))

            print("For model " + str(i+1) +", we compute an AIC of " + str(AIC[-1]) )
        

        #top eqn has lowest AIC
        learned_eqn_index = np.argmin(AIC)
        y_learned = u_sim[learned_eqn_index]
        AIC_learned_eqns = AIC[learned_eqn_index]

        
        #save
        data = {}
        data['final_xi'] = xi_list[learned_eqn_index]
        data['learned_AIC'] = AIC_learned_eqns
        data['description'] = description
        data['u_learned'] = y_learned

        np.save(self.write_dir[0]+"_xi_results_final",data)

        print("Final inferred Equation is of the form : ")
        print(print_pde(data['final_xi'],description))
        
    


    def param_est_FKPP(self):

        '''
        Loads in surface-fit data & derivative files at specified location. 
        Constructs library comprised of FKPP Equation (i.e., theta = [u_{xx} , u(1-u)])
        and infers the two parameters self.real times. Each instance of these parameter instances is saved and recorded.

        Input:
            
        Output:
        
        '''


        #create list of final results
        xi_list = [[] for d in self.data_dir]
        xi_list_no_prune = [[] for d in self.data_dir]
        data_description = [chr(117+i) for i in np.arange(len(self.data_dir))]
        
        #load in data source
        mat = [np.load(d+'.npy', allow_pickle=True, encoding='latin1').item() for d in self.data_dir]
        
        print("Loading in data for PDE-FIND ... ")
    
    
        #load in library and du/dt
        X,T,Ut,Theta,description = theta_construct_FKPP(mat,self.deg)
        
        
        print("Loaded in data for PDE-FIND")
        
        self.description = description
        x = np.unique(X)
        t = np.unique(T)
        
        
        #now perform the training "reals" times
        for r in np.arange(self.reals):
            for i,d in enumerate(self.data_dir):
                
                utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,_,_,_ = data_shuf(Ut[i],Theta,self.shuf_method,self.trainPerc,self.valPerc,len(x),len(t))
                
                xi = np.linalg.lstsq(thetaTrain,utTrain)[0]
                
                xi_list[i].append(xi)

        #save results?
        if self.save_xi == True:
            data = {}
            data['xi_list'] = xi_list[i]
            data['description'] = description
            np.save(self.write_dir[0]+"_param_est_results",data)
