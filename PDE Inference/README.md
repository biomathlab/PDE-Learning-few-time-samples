#PDE Inference

Here we provide the code used for Equation learning and model selection in "Learning Equations from Biological Data with Limited Time Samples" by John T. Nardini, John H. Lagergren, Andrea Hawkins-Daarud, Lee Curtin, Bethan Morris, Erica M. Rutter, Kristin R. Swanson, and Kevin B. Flores.

**PDEFind Exampl.ipynb** is where you likely want to start. Here, we provide an example notebook to document how to use our code and implement it. 

**PDEFind_class.py** provides the PDEFindclass class used for equation inference used in our study.

**PDE_Find2.py** provides some extra PDE Find methods to help with analysis. Some of this code is borrowed or updated from [https://github.com/snagcliffs/PDE-FIND]. 

**model_selection_IP.py** provides code to simulate PDEs for model selection

**DEMO_PDE_FIND_SF_1.py** is example code used to run the Equation learning code for many datasets at once.

**learned_eqn_by_noise_two_col.py** creates tables 2 and 5 from our manuscript.

**param_est-UQ_table.py** creates tables 3 and 4 from our manuscript

**sim_fit_predict_dynamics.py** is used to create figures 5 and 8-11 in our manuscript.

**data/** contains the noisy spatiotemporal data

**surface_data/** contains the denoises data from Data Denoising/

**pickle_data** contains the data from our EQL methodology.
