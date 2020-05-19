import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
import pdb
import math

from sklearn.neighbors import NearestNeighbors

def class_convert_vars(sim,time_int,N,noise):

    """
    Converts the variables sim, time_int, N, noise into corresponding variables to 
    communicate with the PDEFind class

    input:
        sim.        : simulation type (slow,diffuse,fast,nodular)
        time_int    : Simulation timescale (long,short)
        N           : number of time samples (3,5,10)
        noise       : noise level (0.01,0.05)

    output:
        sim.        : simulation number (1,3,2,4)
        time_int    : Simulation timescale ("","_early_time)
        N           : number of time samples ("03","05","10")
        noise       : noise level ("01","05")

    """


    if sim.lower() == "slow":
        sim = 1
    elif sim.lower() == "diffuse":
        sim=3
    elif sim.lower() == "fast":
        sim=2
    elif sim.lower() == "nodular":
        sim=4
    else:
        raise Exception("sim does not match one of the four considered")
        
    if time_int.lower() == "long":
        time_int = ""
    elif sim.lower() == "short":
        time_int = "_early_time"
    else:
        raise Exception("time interval does not match one of the four considered")
        
    if N == 3:
        N = "03"
    elif N == 5:
        N = "05"
    elif N == 10:
        N = "10"
    else:
        raise Exception("N does not match one of the four considered")
        
    if noise == 0.01:
        noise = "01"
    elif noise == 0.05:
        noise = "05"
    else:
        raise Exception("noise does not match one of the four considered")


    return sim,time_int,N,noise

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    n,d = data.shape
    m,d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n,1), dtype=np.complex64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.complex64)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])

    # (Added 9-18-2018 by JTN) Add on derivatives times derivatives
    for D1 in range(derivatives.shape[1]):
        for D2 in range(1,D1+1):
            Theta = np.hstack([Theta, np.multiply(derivatives[:,D1].reshape(n,1),derivatives[:,D2].reshape(n,1))])
            if D1==D2:
                descr.append(derivatives_description[D1] + '^2')
            else:
                descr.append(derivatives_description[D1] + derivatives_description[D2])
                
    return Theta, descr

def print_pde(w, rhs_description, ut = 'u_t',n=5,imag_print=0):

    '''
    records the equation form given as a string.

    Input:
        
        w                   : np.array of equation parameters
        rhs_description     : list of equation terms 
        ut                  : string of what the left hand side looks like
        n                   : precision of printed parameters

    Output:
    
        pde                 : string of equation form

    '''

    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            if imag_print == 0:
                pde = pde + "(%05f)" % (w[i].real) + rhs_description[i] + "\n   "
            else:
                pde = pde + "(%0"+str(n)+"f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    #print(pde)
    return(pde)


def write_pde(w, rhs_description, ut = 'u_t'):

    '''
    records the equation form given as a string.

    Input:
        
        w                   : np.array of equation parameters
        rhs_description     : list of equation terms 
        ut                  : string of what the left hand side looks like
        
    Output:
    
        pde                 : string of equation form

    '''


    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            if w[i].imag == 0:
                pde = pde + "%03.3f" % (w[i].real) + rhs_description[i] + "\n   "
            else:
                pde = pde + "%03.3f %+03.3fi" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)
    return pde


def print_pde_table(w, rhs_description, se = None, ut = 'u_t',n=3):

    '''
    records the equation form given as a string for display in a LaTeX table.

    Input:
        
        w                   : np.array of equation parameters
        rhs_description     : list of equation terms 
        se                  : np.array of standard error terms
        n                   : precision of printed parameters
        ut                  : string of what the left hand side looks like

    Output:
    
        pde                 : string of equation form

    '''


    pde = ut + ' = '
    first = True
    #pdb.set_trace()
    for i in range(len(w)):
        if se is None :
            if w[i] != 0:
                if not first:
                    if w[i].real>0:
                        pde = pde + ' + '        
                pde = pde + str(round((w[i].real),3)) + rhs_description[i]
                first = False
        else:
            if w[i] != 0:
                if not first:
                    pde = pde + ' + '        
                pde = pde + '('+str(round((w[i].real),3))+ '\\pm '+ str(round((se[i].real),4)) +')'+ rhs_description[i]
                first = False
    if pde == ut + ' = ':
        pde = pde + "0"
    return(pde)




##### Functions for sparse regression. #####

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
    
    '''
    Performs Ridge regression with sequential thresholding

    Input:
        
        R                   : matrix with columns representing possible equation terms
        Ut                  : Vector of time derivatives
        lam                 : regularization parameter
        maxit               : number of iterations to perform
        STR_iters           : number of thresholding iterations to perform
        l0_penality         : penalty for nonzero terms
        normalize           : what norm (if any) to normalize by
        split.              : percentage of points to go into training
        print_best_tol      : whether or not to print tolerance

    Output:
    
        w_best              : sparse vector representing inferred equation

    '''


    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=-1)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase lambda until we see decrease in testing performance
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return w_best

def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    

    '''
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1

    Input:
        
        X0                  : matrix with columns representing possible equation terms
        Y                   : Vector of time derivatives
        lam                 : regularization parameter
        w                   : initial guess
        maxit               : number of iterations to perform
        normalize           : what norm (if any) to normalize by
        
    Output:
    
        w              : sparse vector representing inferred equation

    '''

    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y,rcond=-1)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):

    '''
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    

    Input:
        
        X0                  : matrix with columns representing possible equation terms
        Y                   : Vector of time derivatives
        lam                 : regularization parameter
        maxit               : number of iterations to perform
        tol                 : tolerance
        normalize           : what norm (if any) to normalize by
        print_results       : wheter to print results or not
        
    Output:
    
        w              : sparse vector representing inferred equation

    '''


    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=-1)[0]
    else: w = np.linalg.lstsq(X,y,rcond=-1)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=-1)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
    
def FoBaGreedy(X, y, epsilon = 0.1, maxit_f = 1000, maxit_b = 5, backwards_freq = 5):
    
    '''
    Forward-Backward greedy algorithm for sparse regression.

    See Zhang, Tom. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear
    Models', NIPS, 2008

    Input:
        
        X                     : matrix with columns representing possible equation terms
        Y                     : Vector of time derivatives
        epsilon               : tolerance
        maxit_f               : number of forward iterations to perform
        maxit_b               : number of backward iterations to perform per step
        backwards_freq        : how frequently to perform backward iteration
        
    Output:
    
        w                     : sparse vector representing inferred equation

    '''


    n,d = X.shape
    F = {}
    F[0] = set()
    w = {}
    w[0] = np.zeros((d,1), dtype=np.complex64)
    k = 0
    delta = {}

    for forward_iter in range(maxit_f):

        k = k+1

        # forward step
        non_zeros = np.where(w[k-1] == 0)[0]
        err_after_addition = []
        residual = y - X.dot(w[k-1])
        for i in range(len(non_zeros)):
            alpha = X[:,i].T.dot(residual)/np.linalg.norm(X[:,i])**2
            w_added = np.copy(w[k-1])
            w_added[i] = alpha
            err_after_addition.append(np.linalg.norm(X.dot(w_added)-y))
        i = np.argmin(err_after_addition)
        
        F[k] = F[k-1].union({i})
        w[k] = np.zeros((d,1), dtype=np.complex64)
        w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y,rcond=-1)[0]

        # check for break condition
        delta[k] = np.linalg.norm(X.dot(w[k-1]) - y) - np.linalg.norm(X.dot(w[k]) - y)
        if delta[k] < epsilon: return w[k-1]

        # backward step, do once every few forward steps
        if forward_iter % backwards_freq == 0 and forward_iter > 0:

            for backward_iter in range(maxit_b):

                non_zeros = np.where(w[k] != 0)
                err_after_simplification = []
                for j in range(len(non_zeros)):
                    w_simple = np.copy(w[k])
                    w_simple[j] = 0
                    err_after_simplification.append(np.linalg.norm(X.dot(w_simple) - y))
                j = np.argmin(err_after_simplification)
                w_simple = np.copy(w[k])
                w_simple[j] = 0

                # check for break condition on backward step
                delta_p = err_after_simplification[j] - np.linalg.norm(X.dot(w[k]) - y)
                if delta_p > 0.5*delta[k]: break

                k = k-1;
                F[k] = F[k+1].difference({j})
                w[k] = np.zeros((d,1), dtype=np.complex64)
                w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y,rcond=-1)[0]

    return w[k] 
    
##### functions added by JTN

def binshuffle (xbin,tbin,xn,tn,trainPerc=0.6,valPerc=0.2):
    
    '''
    creates indices for training, validation, and testing data. 
    Assumes data is on mesh grid of x and t values

    Input:

        xbin            : size of x bins
        tbin            : size of t bins
        xn              : total number of x points
        tn              : total number of t points
        trainPerc       : percentage of training points
        valPerc         : percentage of validation points
        
        
        
    Output:
    
        ptrain          : indices for training points
        pval            : indices for validation points
        ptest           : indices for testing points
        

    '''

    #number t,x and all bins
    Nt = np.ceil(np.float(tn)/tbin)
    Nx = np.ceil(np.float(xn)/xbin)
    Nbins = int(Nt*Nx)
    #permute these bins
    pbin = np.random.permutation(Nbins)
    
    #split (xbin \times tbin) bins into train, val, and test bins
    train_pbin = pbin[:int(Nbins*trainPerc)]
    val_pbin = pbin[int(Nbins*trainPerc):int(Nbins*(trainPerc+valPerc))]
    test_pbin = pbin[int(Nbins*(trainPerc+valPerc)):]
    
    #initialize indices
    #vector of all points from 0 to xn*tn-1
    ind = np.arange(xn*tn)
    indModtn = np.mod(ind,tn)
    
    #indModxn = np.mod(ind,xn)
    
    
    #training indices
    ptrain = []
    for p in train_pbin:
        #t bin index 
        pmodt = np.mod(p,Nt)
        #x bin index
        pmodx = np.floor(p/Nt)
        
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
        
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        #add to train indices
        ptrain = np.concatenate((ptrain,new_ind[0]),axis=0)

    #validation indices
    pval = []
    for p in val_pbin:
        
        
        #t bin index 
        pmodt = np.mod(p,Nt)
        #x bin index
        pmodx = np.floor(p/Nt)
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
        '''ind_loc = ([pmodx*xbin <= indModxn , indModxn < (pmodx+1)*xbin ,
                   pmodt*xn*tbin <= ind, ind < (pmodt+1)*xn*tbin])'''
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        
        #add to validation indices
        pval = np.concatenate((pval,new_ind[0]),axis=0)
        
    #test indices
    ptest = []
    for p in test_pbin:
        #t bin index 
        pmodt = int(np.floor(p/Nx))
        #x bin index
        pmodx = np.mod(p,Nx)
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
        
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        
        #add to test indices
        ptest = np.concatenate((ptest,new_ind[0]),axis=0)

    ptrain = ptrain.astype(int)
    pval = pval.astype(int)
    if ptest!=[]: ptest = ptest.astype(int)

    return ptrain, pval, ptest

#shuffle the data in train, val, test data
def data_shuf(Ut,R,shufMethod,trainPerc,valPerc,xn,tn,xbin=5,tbin=5):

    '''
    shuffles data into training, validation, and test sets.

    Input:

        Ut              : vector of time derivatives
        R               : matrix of potential equation terms
        shufMethod      : how to shuffle data
        trainPerc       : percent of training data
        valPerc         : percent of validation data data
        xn              : number of x points
        tn              : number of t points
        xbin            : width of x bins (if binning data)
        tbin            : width of t bins (if binning data)
        
        
        
    Output:
    
        UtTrain         : Training data for Ut
        RTrain          : Training data for R
        ptrain          : training indices
        UtVal           : Validation data for Ut
        RVal            : Validation data for R
        pval            : Validation indices
        UtTest          : Testing data for Ut
        RTest           : Validation data for R
        ptest           : Testing indices
        

    '''
    
    p_length = len(Ut)
    
    #permute
    if shufMethod == 'perm':
        p = np.random.permutation(p_length)
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]

        
    #do not permute
    elif shufMethod == 'noperm':
        p = np.arange(p_length)
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]
        
    #do not permute, take last time points first
    elif shufMethod == 'reverse':
        p = np.squeeze(np.fliplr(np.atleast_2d(np.arange(p_length))))
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]
        
    elif shufMethod == 'bins':
        ptrain,pval,ptest = binshuffle(xbin,tbin,xn,tn,trainPerc,valPerc)
        p = np.concatenate((ptrain,pval,ptest))
                
    ntrain = len(ptrain)
    nval = len(pval)
    ntest = len(ptest)
    
    #split into train, val, test
    UtTrain = Ut[ptrain]
    RTrain = R[ptrain,:]
    
    UtVal = Ut[pval]
    RVal = R[pval,:]
    
    UtTest = Ut[ptest]
    RTest = R[ptest,:]
    
    
    return UtTrain, RTrain, ptrain, UtVal, RVal, pval, UtTest, RTest, ptest 



def run_PDE_Find_train_val (RTrain,UtTrain,RVal,utVal,algoName,description,lambda_lb=-2,lambda_ub=3,deriv_list=None):
        
    '''
    Performs hyperparameter selection

    Input:
        
        RTrain              : Training matrix with columns representing possible equation terms
        UtTrain             : Training Vector of time derivatives
        RVal                : Validation matrix with columns representing possible equation terms
        UtVal               : Validation Vector of time derivatives
        algoName            : string of method to use ("Lasso", "STRidge", or "Greedy")
        description         : list of strings representing potential equation terms 
        lambda_lb           : lower bound for hyperparameters (will be raised 10^lambda_lb)
        lambda_ub           : upper bound for hyperparameters (will be raised 10^lambda_ub)
        deriv_list          : list of true underlying terms
    
    Output:
    
        xi_best             : final inferred vector
        hparams_opt         : list of considered hyperparameters
        val_score           : validation scores at these hyperparameters
        TP_FN_score         : list of either TPR scores or number of nonzero terms

    '''


    
    if algoName == 'STRidge':
        

        lambda_vec = np.hstack([0,10**np.linspace(lambda_lb,lambda_ub,20)])
        dtol_vec =  np.hstack([0,10**np.linspace(lambda_lb,lambda_ub,20)])
        

        # convert to mesh
        [l_mesh,dtol_mesh] = np.meshgrid(lambda_vec,dtol_vec)
        # create ordered pairs
        l_mesh = l_mesh.reshape(-1)[:,np.newaxis]
        dtol_mesh = dtol_mesh.reshape(-1)[:,np.newaxis]
        X_mesh = np.concatenate([l_mesh,dtol_mesh],axis=1)

        val_score = np.zeros(len(X_mesh))
        TP_FN_score = np.zeros(len(X_mesh))
        
        for i,hparams in enumerate(X_mesh):
            xi = STRidge(RTrain,UtTrain,hparams[0], 1000, hparams[1])
            val_score[i] = run_PDE_Find_Test(RVal,utVal,xi)
        if deriv_list != None:
            TP_FN_score[i] = TP_TPFPFN(xi,description,deriv_list,1e-4)
        else:
            #just count number of nonzero term
            TP_FN_score[i] = np.count_nonzero(xi)

        #find optimal hyper parameter
        hparams_opt = X_mesh[np.argmin(val_score),]
        #re-find best estimate
        xi_best = STRidge(RTrain,UtTrain,hparams_opt[0], 1000, hparams_opt[1])
        
        hparams_opt = X_mesh
        
    elif algoName == 'Greedy':
        
        lambda_vec = np.hstack([0,10**np.linspace(lambda_lb,lambda_ub,30)])

        val_score = np.zeros(len(lambda_vec))
        TP_FN_score = np.zeros(len(lambda_vec))
        
        for j,l in enumerate(lambda_vec):
            xi = FoBaGreedy(RTrain, UtTrain,l)
            
            val_score[j] = run_PDE_Find_Test(RVal,utVal,xi)
            if deriv_list != None:
                TP_FN_score[j] = TP_TPFPFN(xi,description,deriv_list,1e-4)
            else:
                #just count number of nonzero term
                TP_FN_score[j] = np.count_nonzero(xi)
            
        #find optimal hyperparameter
        hparams_opt = lambda_vec[np.argmin(val_score)]
        #re-find best estimate
        xi_best = FoBaGreedy(RTrain, UtTrain,hparams_opt)
        
    elif algoName == 'Lasso':
        
        lambda_vec = np.hstack([0,10**np.linspace(lambda_lb,lambda_ub,50)])
        
        val_score = np.zeros(len(lambda_vec))
        TP_FN_score = np.zeros(len(lambda_vec))
        
        for j,l in enumerate(lambda_vec):
            xi = Lasso(RTrain, UtTrain, l, normalize = 1,maxit = 100)
        
            val_score[j] = run_PDE_Find_Test(RVal,utVal,xi)
            
            if deriv_list != None:
                TP_FN_score[j] = TP_TPFPFN(xi,description,deriv_list,1e-4)
            else:
                #just count number of nonzero term
                TP_FN_score[j] = np.count_nonzero(xi)
                
        #find optimal hyperparameter
        hparams_opt = lambda_vec[np.argmin(val_score)]
        #re-find best estimate
        xi_best = Lasso(RTrain, UtTrain, hparams_opt, normalize = 1,maxit = 100)
        

    return xi_best, hparams_opt, val_score, TP_FN_score

#Test  the trained w value
def run_PDE_Find_Test (RTest,UtTest,w):
   
    '''
    calulate || Rw - Ut ||_2

    Input:
        
        RTest           : Matrix of potential equation terms
        UtTest          : vector of time derivatives
        w               : vector if inferred equation form
    Output:
    
        score           : MSE between Rw and Ut

    '''

    test_length = len(RTest)
    #Compute test score
    score = np.linalg.norm(UtTest - np.matmul(RTest,w))/test_length
    
    return score

def TP_TPFPFN (w,rhs_des,deriv_list,thres=1e-4):

    '''
    Calulate TP = TP / (TP + FN + FP)

    Input:
        
        w               :  vector representing inferred equation
        rhs_des         :  list of equation terms
        deriv_list      : list of true terms in underlying model
        thres           : threshold at which parameter considered zero.

    Output:
    
        TPR           : TPR score

    '''


    
    #zero gets a score of 0
    if len(w) == 0:
        return 0.0
    
    #initialize   
    TP = 0
    FP = 0
    FN = 0

    deriv_list_c = deriv_list[:]
    
    remove_from_deriv_list = []
    #account for any deriv_list entries not in rhs_des
    for i in deriv_list_c:
        if i not in rhs_des:
            FN+=1
            remove_from_deriv_list.append(deriv_list_c.index(i))
            
    #remove in reverse direction
    for i in sorted(remove_from_deriv_list,reverse=True):
        del deriv_list_c[i]
    
    
    #find their locations in w
    deriv_inds = [rhs_des.index(j) for j in deriv_list_c]
    
    #update TP, FN    
    for i in deriv_inds:
        if np.abs(w[i]) > thres:
            TP += 1
        else:
            FN += 1
    
    #remove deriv_inds from consideration
    all_ind = np.arange(len(w))
    extra_ind = np.delete(all_ind,deriv_inds)
    
    #loop through remaining terms, penalize if not zero
    for i in extra_ind:
        if np.abs(w[i]) > thres:
            FP += 1
            
    return(float(TP)/float(TP + FP + FN))


def theta_construct_1d(mat,deg):
    
    '''
    Take in mat files and build library of terms for PDE-FIND
    
    inputs:
    mat : list of lists with two entries: "variables" and "variable_names". Each list corresponds to a dependent variables 
    deg : degree of polynomials to consider in PDE-FIND library
    
    outputs:
    X : Vector of X-values corresponding to the rows of Theta
    T : Vector of T-values corresponding to the rows of Theta
    Ut : List of computed du/dt values
    Theta : Matrix comprising the library of terms for PDE-FIND
    description : list whose entries correspond to the columns in Theta
    '''
    
       
    #assuming all mar files have same x,y dims
    X = mat[0]['variables'][0][:,np.newaxis]
    T = mat[0]['variables'][1][:,np.newaxis]
        
    #U values
    U = [m['variables'][2] for m in mat]
    Ux = [m['variables'][3][:,np.newaxis] for m in mat]
    Ut = [m['variables'][4][:,np.newaxis] for m in mat]
    Uxx = [m['variables'][5][:,np.newaxis] for m in mat]
    
    num_points = len(U[0])
  
    #format strings for description
    data_description = [chr(117+i) for i in np.arange(len(mat))]
    X_ders_descr = [""]
    X_ders_descr_x = [chr(117+i)+'_{x}' for i in np.arange(len(mat))]
    X_ders_descr_xx = [chr(117+i)+'_{xx}' for i in np.arange(len(mat))]
    X_ders_descr = X_ders_descr + X_ders_descr_x + X_ders_descr_xx
    
    # Form a huge matrix using up to quadratic polynomials in all variables.
    X_data = np.hstack(tuple(U))
    X_ders = np.hstack((np.ones((num_points,1)),np.hstack(tuple(Ux)),np.hstack(tuple(Uxx))))

    Theta, description = build_Theta(X_data, X_ders, X_ders_descr, deg, data_description = data_description)

    Theta = np.hstack([Theta,U[0]*Ux[0]**2])
    description.append('uu_{x}^2')

    #Remove terms that don't make modeling sense or can't conserve
    #mass
    Theta = np.delete(Theta,description.index(""),axis=1)
    description.remove("")
    Theta = np.delete(Theta,description.index("u_{xx}^2"),axis=1)
    description.remove("u_{xx}^2")
    Theta = np.delete(Theta,description.index("u_{xx}u_{x}"),axis=1)
    description.remove("u_{xx}u_{x}")
    

    return X,T,Ut,Theta,description

def theta_construct_FKPP(mat,deg):
    
    '''
    Take in mat files and build library of terms for PDE-FIND
    
    inputs:
    mat : list of lists with two entries: "variables" and "variable_names". Each list corresponds to a dependent variables 
    deg : degree of polynomials to consider in PDE-FIND library
    
    outputs:
    X : Vector of X-values corresponding to the rows of Theta
    T : Vector of T-values corresponding to the rows of Theta
    Ut : List of computed du/dt values
    Theta : Matrix comprising the library of terms for PDE-FIND
    description : list whose entries correspond to the columns in Theta
    '''
    
       
    #assuming all mar files have same x,y dims
    X = mat[0]['variables'][0][:,np.newaxis]
    T = mat[0]['variables'][1][:,np.newaxis]
        
    #U values
    U = [m['variables'][2] for m in mat]
    Ut = [m['variables'][4][:,np.newaxis] for m in mat]
    Uxx = [m['variables'][5][:,np.newaxis] for m in mat]
    
    #sample_ind = np.where(U[0]>=0)[0]

    Ut = [Ut[0]]
    Theta = np.hstack([U[0]*(1-U[0]),Uxx[0]])
    description = ['u(1-u)','u_{xx}']

    return X,T,Ut,Theta,description

def PDE_FIND_prune_lstsq(xi,utTrain,utVal,thetaTrain,thetaVal,description,val_score_init,prune_level):
    
    '''
    perform pruning
    
    inputs:
    
    xi
    utTrain              : Training Vector of time derivatives
    utVal                : Validation Vector of time derivatives
    thetaTrain           : Training matrix with columns representing possible equation terms
    thetaVal             : Validation matrix with columns representing possible equation terms
    description          : List of potential equation terms
    val_score_init       : validation score for xi
    prune_level          : pruning threshold
   


    outputs:
    
    xi_tilde             : pruned xi vector 
    val_score_new        : validation score for xi parameter

    '''
    


    xi_tilde = xi[xi!=0]
    thetaTrain_tilde = thetaTrain[:,np.squeeze(xi!=0)]
    thetaVal_tilde = thetaVal[:,np.squeeze(xi!=0)]
    description_tilde = [description[i] for i in np.where(np.squeeze(xi!=0))[0]]

    #loop through each remaining entry and see how much the validation score changes when this entry is set to zero
    val_score_vec = np.zeros(np.squeeze(xi_tilde.shape))
    for i in np.argsort(np.abs(xi_tilde)):
        #copy xi and theta without ith entry in xi_hat, theta_hat 
        thetaTrain_hat = np.delete(thetaTrain_tilde,i,1)
        thetaVal_hat = np.delete(thetaVal_tilde,i,1)
        
        #nonsparse regression
        xi_hat = np.linalg.lstsq(thetaTrain_hat,utTrain,rcond=-1)[0]#run_PDE_Find_train(thetaTrain_hat,utTrain,algoName,0)
        #get new validation score
        val_score_new = run_PDE_Find_Test(thetaVal_hat,utVal,xi_hat)
        val_score_vec[i] = val_score_new
    
    #indices of params to keep, keep if validation score goes up more than prune_le vel %
    keep_ind = np.squeeze(val_score_vec >= (1+prune_level)*val_score_init)
    if np.sum(keep_ind) <= 1:
        keep_ind = np.squeeze(val_score_vec == np.max(val_score_vec))
    #final theta matrices
    thetaTrain_tilde = thetaTrain_tilde[:,keep_ind]
    thetaVal_tilde = thetaVal_tilde[:,keep_ind]
    description_tilde = [description_tilde[i] for i in np.where(np.squeeze(keep_ind))[0]]
    #final training on final library
    if thetaTrain_tilde.shape[1] > 0:
        xi_tilde = np.linalg.lstsq(thetaTrain_tilde,utTrain,rcond=-1)[0]#xi_tilde[keep_ind]
    else:
        xi_tilde = []
    #calculate final validation score
    val_score_new = run_PDE_Find_Test(thetaVal_tilde,utVal,xi_tilde)
       
    # re-convert xi_tilde to same length as initial description
    xi_tilde = xi_convert_full(xi_tilde,description_tilde,description)
    
   
    return xi_tilde, val_score_new
        
def trans(x,N):

    '''
    convert decimal number to binary representation
    
    inputs:
    
    x           : number
    N           : length of binary representation
   
    outputs:

    w           : binary vector from x
    '''

    y=np.copy(x)
    if y == 0: return[0]
    bit = []
    for i in np.arange(N):
        bit.append(y % 2)
        y >>= 1
    return np.atleast_2d(np.asarray(bit[::-1]))

#go from binary representation to decimal number
def trans_rev(x):

    '''
    convert binary representation to decimal number
    
    inputs:
    
    x           : binary vector 
   
    outputs:

    dec         : decimal number
    '''


    n = len(x)-1
    dec = 0
    for i in np.arange(n+1):
        dec = dec + x[i]*2**(n-i)
    return dec

def most_common(lst):
    '''
    find most common entry from list
    
    inputs:
    
    lst           : list
   
    outputs:

    count of list terms.
    '''


    return max(lst, key=lst.count)


def xi_convert_full(xi,desc,desc_full):

    '''
    convert partial xi from partial library to full xi for full  library
    
    inputs:
    
    xi          : vector corresponding to partial list
    desc        : partial list of equation terms
    desc_full   : full list of equation terms
   
    outputs:

    desc_full   : vector corresponding to full list
    '''


    xi_full = np.zeros((len(desc_full),1))
    
    for i,x in enumerate(xi):
        #where is this ith term located in full description
        desc_ind = desc_full.index(desc[i])
        
        xi_full[desc_ind] = np.real(xi[i])
    
    
    return xi_full

def create_true_xi(description,deriv_index,params):

    '''
    create vector with relevant terms and parameters
    
    inputs:
    
    description     : list of equation terms
    deriv_index     : list of equation terms in true model
    params          : list of parameters for these terms.

    outputs:

    xi_true         : vector corresponding to true equation form.
    '''


    xi_true = np.zeros((len(description),1))
    
    for i in np.arange(len(deriv_index)):
        xi_true[description.index(deriv_index[i])] = params[i]

    return xi_true
