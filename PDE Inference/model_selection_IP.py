import numpy as np

from scipy import integrate
from scipy import sparse
from scipy import interpolate

import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb


def create_Ux_mat(x):

    """
    create u_{x} matrix operator using the upwind method.

    input:
        x: numerical grid

    returns:
        A: Sparse Matrix such that A.dot(u) approximates u_{x} for 1d nparray u
    """
    
    dx = x[1] - x[0]
    
    ## upwind method based on left cells moving right
    ## and right cells moving left (similar to a scratch assay)
    '''x_int = np.arange(1,len(x)-1,dtype=int)
    Ux_mat_row = np.hstack((x_int, x_int))
    Ux_mat_col = np.hstack((x_int, x_int[:len(x_int)/2]-1,x_int[len(x_int)/2:]+1))
    Ux_entry = (1/dx)*np.hstack((np.ones(len(x_int)/2),-np.ones(len(x_int)/2),-np.ones(len(x_int)/2),np.ones(len(x_int)/2)))
    
    Ux_mat_row_bd = np.hstack((0,0,len(x)-1,len(x)-1))
    Ux_mat_col_bd = np.hstack((0,1,len(x)-2,len(x)-1))
    Ux_entry_bd = (1/dx)*np.hstack((-1,1,-1,1))

    Ux_mat_row = np.hstack((Ux_mat_row,Ux_mat_row_bd))
    Ux_mat_col = np.hstack((Ux_mat_col,Ux_mat_col_bd))
    Ux_entry = np.hstack((Ux_entry,Ux_entry_bd))'''

    ## upwind method based on right cells (x>0) moving right
    ## and left cells (x<0) moving left (expansion)
    x_int = np.arange(1,len(x)-1,dtype=int)
    Ux_mat_row = np.hstack((x_int, x_int))

    Ux_mat_col = np.hstack((x_int, x_int[:len(x_int)//2]+1,x_int[len(x_int)//2:]-1))
    Ux_entry = (1/dx)*np.hstack((-np.ones(len(x_int)//2),np.ones(len(x_int)//2),np.ones(len(x_int)//2),-np.ones(len(x_int)//2)))
    
    Ux_mat_row_bd = np.hstack((0,0,len(x)-1,len(x)-1))
    Ux_mat_col_bd = np.hstack((0,1,len(x)-2,len(x)-1))
    Ux_entry_bd = (1/dx)*np.hstack((-1,1,-1,1))

    Ux_mat_row = np.hstack((Ux_mat_row,Ux_mat_row_bd))
    Ux_mat_col = np.hstack((Ux_mat_col,Ux_mat_col_bd))
    Ux_entry = np.hstack((Ux_entry,Ux_entry_bd))

    return sparse.coo_matrix((Ux_entry,(Ux_mat_row,Ux_mat_col)))


def create_Uxx_mat(x):

    """
    create u_{xx} matrix operator using central differences.

    No flux boundaries assumed.

    input:
        x: numerical grid

    returns:
        A: Sparse Matrix such that A.dot(u) approximates u_{xx} for 1d nparray u
    """
    

    dx = x[1] - x[0]
    
    x_int = np.arange(1,len(x)-1,dtype=int)
    #create u_{xx} matrix operator
    Uxx_mat_row = np.hstack((x_int,x_int,x_int))
    Uxx_mat_col = np.hstack((x_int-1,x_int,x_int+1))
    Uxx_entry = (1/(dx**2))*np.hstack((np.ones(len(x)-2),-2*np.ones(len(x)-2),(np.ones(len(x)-2))))

    Uxx_mat_row_bd = np.hstack((0,0,len(x)-1,len(x)-1))
    Uxx_mat_col_bd = np.hstack((0,1,len(x)-2,len(x)-1))
    Uxx_entry_bd = (1/(dx**2))*np.hstack((-2,2,2,-2))


    Uxx_mat_row = np.hstack((Uxx_mat_row,Uxx_mat_row_bd))
    Uxx_mat_col = np.hstack((Uxx_mat_col,Uxx_mat_col_bd))
    Uxx_entry = np.hstack((Uxx_entry,Uxx_entry_bd))

    return sparse.coo_matrix((Uxx_entry,(Uxx_mat_row,Uxx_mat_col)))

def create_Uxx_mat_2d(dx,x_int,x_bd_0,x_bd_l,step=1):

    #interior stencil
    Uxx_mat_row = np.hstack((x_int,x_int,x_int))
    Uxx_mat_col = np.hstack((x_int-step,x_int,x_int+step))
    Uxx_entry = (1/(dx**2))*np.hstack((
        np.ones(x_int.shape),-2*np.ones(x_int.shape),
        np.ones(x_int.shape)
        ))

    #boundary stencil
    Uxx_mat_row_bd = np.hstack((x_bd_0,x_bd_0,x_bd_l,x_bd_l))
    Uxx_mat_col_bd = np.hstack((x_bd_0,x_bd_0+step,x_bd_l-step,x_bd_l))
    Uxx_entry_bd = (1/(dx**2))*np.hstack((
        -2*np.ones(x_bd_0.shape),2*np.ones(x_bd_0.shape),
        2*np.ones(x_bd_l.shape),-2*np.ones(x_bd_l.shape)
        ))

    #put it all together
    Uxx_mat_row = np.hstack((Uxx_mat_row,Uxx_mat_row_bd))
    Uxx_mat_col = np.hstack((Uxx_mat_col,Uxx_mat_col_bd))
    Uxx_entry = np.hstack((Uxx_entry,Uxx_entry_bd))

    return sparse.coo_matrix((Uxx_entry,(Uxx_mat_row,Uxx_mat_col)))


def g(y):

    """
    baseline diffusion function.

    input:
        y: numerical grid

    returns:
        f: nparray of ones with the same size as y
    """
    

    return np.ones(y.shape)
def f(y):

    """
    baseline proliferation function (logistic growth)

    input:
        y: numerical grid

    returns:
        f: nparray of y*(1-y)
    """
    

    k = 1.0
    return y*(1-y)

def PDE_RHS_FKPP(t,y,q,x,description=None):

    """
    RHS for the Fisher-KPP Equation

    input:

        t: time
        y: PDE solution
        q: parameters D and r
        x: space grid
        description: (not necessary) potential learned terms

    returns:
        RHS: nparray of RHS at time t.
    """
    
    
    D_mat = create_Uxx_mat(x)

    return  (q[0]*D_mat.dot(y) + 
             q[1]*y*(1-y))  

def learned_RHS(t,y,q,x,desc):

    """
    RHS for an inferred Equation

    input:

        t: time
        y: PDE solution
        q: inferred params from PDE-Find
        x: space grid
        desc: Library of potential learned terms

    returns:
        RHS: nparray of RHS at time t.
    """
    
    
    Ux_mat = create_Ux_mat(x)
    Uxx_mat = create_Uxx_mat(x)

    return (q[desc.index('u_{x}')]*Ux_mat.dot(y) + 
            q[desc.index('u_{xx}')]*Uxx_mat.dot(y) +
            q[desc.index('u^2')]*y**2 +
            q[desc.index('u')]*y + 
            q[desc.index('u^2u_{x}')]*(y**2)*Ux_mat.dot(y) +  
            q[desc.index('uu_{x}')]*y*Ux_mat.dot(y) +   
            q[desc.index('u^2u_{xx}')]*(y**2)*Uxx_mat.dot(y) + 
            q[desc.index('uu_{xx}')]*y*Uxx_mat.dot(y) + 
            q[desc.index('u_{x}^2')]*Ux_mat.dot(y)**2) 
    
    

    

def RSS_GLS(model,target,gamma):

    """
    Generalized least squares error || (y-f)/(f^gamma) ||_2^2 . 
    where f is model , y is data
    
    Because of singularity at f=0 , we use MSE for |f| < 1e-4

    input:

        model: inferred model of data
        target: data points
        gamma: gamma from statistical model
        
    returns:
        loss: squared error between model and target
    """
    
    
    GLS_domain = np.where(np.abs(model)>1e-4)
    OLS_domain = np.where(np.abs(model)<=1e-4)

    GLS_res = (model[GLS_domain]-target[GLS_domain])/(np.abs(model[GLS_domain])**gamma)
    OLS_res = model[OLS_domain]-target[OLS_domain]

    GLS_RSS = np.linalg.norm(GLS_res)**2
    OLS_RSS = np.linalg.norm(OLS_res)**2

    return GLS_RSS + OLS_RSS

def PDE_sim(q,RHS,x,t,IC,f=f,g=g,description=None):

    """
    
    Simulate 1d inferred PDE models using the method of lines approach

    input:

        q:              vector of inferred parameters
        RHS:            PDE to simulate
        x:              Spatial grid
        t:              Time grid
        IC:             Initial condition
        f:              growth rate
        g:              Diffusion rate
        description:    List of potential right hand side PDE terms.
        
    returns:
        y:              PDE solution at (x,t) points
    """

    #grids for numerical integration
    t_sim = np.linspace(t[0],t[-1],10000)
    x_sim = np.linspace(x[0],x[-1],200)
    
    #interpolate initial condition to new grid

    f_interpolate = interpolate.interp1d(x,IC)
    y0 = f_interpolate(x_sim)
        
    #indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    #make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,q,x_sim,description)
            
    y = np.zeros((len(x),len(t)))   # array for solution
    y[:, 0] = IC
    write_count = 0

    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        #write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            f_interpolate = interpolate.interp1d(x_sim,r.integrate(t_sim[i]))
            y[:,write_count] = f_interpolate(x)
        else:
            #otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)
            #raise RuntimeError("Could not integrate")

    return y

