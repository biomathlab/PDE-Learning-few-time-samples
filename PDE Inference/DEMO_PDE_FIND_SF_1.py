from PDEFind_class import PDE_Findclass
import glob, os

dim = 1

datasets = [[os.path.basename(x)[:-4]] for x in glob.glob("data/*"+str(dim)+"d_1*")]
#datasets += [[os.path.basename(x)[:-4]] for x in glob.glob("data/*"+str(dim)+"d_early_time_1*")]


# form of denoising
model_name = 'ann'

#infer multiple equations using PDE-FIND
train_PDEFind = 1

#find top equations from PDE-FIND Results
find_eqns = 0

#Perform model selection for top equations
simulate_learned_compare = 1

#Perform post-processing, final parameter estimation
param_est_final = 0


#estimate parameters for Fisher-KPP model
param_est_FKPP = 0


#training split for PDE-FIND
trainPerc = 0.5
valPerc = 0.5

shuf_method = 'perm'
algo_name = "Greedy"
prune_level = 0.05
num_eqns = 3
save_learned_eqns = True
print_pdes = True
#save all xi values during training
save_xi = True
#save the final value of xi after training
save_learned_xi = True

#PDE-FIND values
data_dir = "surface_data/"



if dim == 1:
	reals = 100
elif dim == 2:
	reals = 1


for dataset in datasets:


	comp_str = "_" + model_name

	pf = PDE_Findclass(dataset,
                       comp_str,
                       data_dir=data_dir,
                       reals=reals,
                       trainPerc = trainPerc,
                       valPerc = valPerc,
                       print_pdes = print_pdes,
                       save_xi = save_xi,
                       save_learned_xi = save_learned_xi,
                       prune_level = prune_level,
                       num_eqns=num_eqns,
                       save_learned_eqns = save_learned_eqns,
                       shuf_method = shuf_method,
                       algo_name = algo_name)
    

	# run options
	if train_PDEFind == 1:
		pf.train_val_PDEFind()
	    
	if find_eqns == 1:
		pf.list_common_eqns()

	if simulate_learned_compare == 1:
		pf.simulate_learned_eqns_compare()

	if param_est_final == 1:
		pf.param_est_final()

	if param_est_FKPP == 1:
		pf.param_est_FKPP()

	

