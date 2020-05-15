import sys, importlib
sys.path.append('../../')

from Modules.Utils.Imports import *

from Modules.Models.BuildSurfaceFitter import BuildSurfaceFitter
from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Activations.Gaussian import Gaussian
from Modules.Losses.WGLSLoss import WGLSLoss
from Modules.Losses.HPRegressionLoss import HPRegressionLoss
from Modules.Losses.HOTRegularization import HOTRegularization
from Modules.Losses.HardBoundingBoxLoss import HardBoundingBoxLoss
from Modules.Utils.Gradient import Gradient
from Modules.Utils.ModelWrapper import ModelWrapper

import argparse

#
# parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("--file_idx", type=int, default=0, help="file index")
parser.add_argument("--verbosity", type=int, default=1, help="")
parser.add_argument("--num_reps", type=int, default=1, help="")
parser.add_argument("--gpu", type=int, default=3, help="")
options = parser.parse_args()
print(options)

#
# pick a gpu
#

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

#
# load data
#

# 1D Fisher KPP
path = '../../Data/Surfaces/FKPP1D/'
#files = glob.glob(path+'fkpp_1d_param_sweep*.npy')
#files += glob.glob(path+'fkpp_1d_early_time_param_sweep*.npy')
#files = [os.path.basename(file) for file in files]
#files.sort()
files = [
    'fkpp_1d_1_03pts_01.npy','fkpp_1d_1_03pts_05.npy','fkpp_1d_1_03pts_10.npy',
    'fkpp_1d_1_05pts_01.npy','fkpp_1d_1_05pts_05.npy','fkpp_1d_1_05pts_10.npy',
    'fkpp_1d_1_10pts_01.npy','fkpp_1d_1_10pts_05.npy','fkpp_1d_1_10pts_10.npy',
    'fkpp_1d_early_time_1_03pts_01.npy','fkpp_1d_early_time_1_03pts_05.npy','fkpp_1d_early_time_1_03pts_10.npy',
    'fkpp_1d_early_time_1_05pts_01.npy','fkpp_1d_early_time_1_05pts_05.npy','fkpp_1d_early_time_1_05pts_10.npy',
    'fkpp_1d_early_time_1_10pts_01.npy','fkpp_1d_early_time_1_10pts_05.npy','fkpp_1d_early_time_1_10pts_10.npy',
    'fkpp_1d_2_03pts_01.npy','fkpp_1d_2_03pts_05.npy','fkpp_1d_2_03pts_10.npy',
    'fkpp_1d_2_05pts_01.npy','fkpp_1d_2_05pts_05.npy','fkpp_1d_2_05pts_10.npy',
    'fkpp_1d_2_10pts_01.npy','fkpp_1d_2_10pts_05.npy','fkpp_1d_2_10pts_10.npy',
    'fkpp_1d_early_time_2_03pts_01.npy','fkpp_1d_early_time_2_03pts_05.npy','fkpp_1d_early_time_2_03pts_10.npy',
    'fkpp_1d_early_time_2_05pts_01.npy','fkpp_1d_early_time_2_05pts_05.npy','fkpp_1d_early_time_2_05pts_10.npy',
    'fkpp_1d_early_time_2_10pts_01.npy','fkpp_1d_early_time_2_10pts_05.npy','fkpp_1d_early_time_2_10pts_10.npy',
    'fkpp_1d_3_03pts_01.npy','fkpp_1d_3_03pts_05.npy','fkpp_1d_3_03pts_10.npy',
    'fkpp_1d_3_05pts_01.npy','fkpp_1d_3_05pts_05.npy','fkpp_1d_3_05pts_10.npy',
    'fkpp_1d_3_10pts_01.npy','fkpp_1d_3_10pts_05.npy','fkpp_1d_3_10pts_10.npy',
    'fkpp_1d_early_time_3_03pts_01.npy','fkpp_1d_early_time_3_03pts_05.npy','fkpp_1d_early_time_3_03pts_10.npy',
    'fkpp_1d_early_time_3_05pts_01.npy','fkpp_1d_early_time_3_05pts_05.npy','fkpp_1d_early_time_3_05pts_10.npy',
    'fkpp_1d_early_time_3_10pts_01.npy','fkpp_1d_early_time_3_10pts_05.npy','fkpp_1d_early_time_3_10pts_10.npy',
    'fkpp_1d_4_03pts_01.npy','fkpp_1d_4_03pts_05.npy','fkpp_1d_4_03pts_10.npy',
    'fkpp_1d_4_05pts_01.npy','fkpp_1d_4_05pts_05.npy','fkpp_1d_4_05pts_10.npy',
    'fkpp_1d_4_10pts_01.npy','fkpp_1d_4_10pts_05.npy','fkpp_1d_4_10pts_10.npy',
    'fkpp_1d_early_time_4_03pts_01.npy','fkpp_1d_early_time_4_03pts_05.npy','fkpp_1d_early_time_4_03pts_10.npy',
    'fkpp_1d_early_time_4_05pts_01.npy','fkpp_1d_early_time_4_05pts_05.npy','fkpp_1d_early_time_4_05pts_10.npy',
    'fkpp_1d_early_time_4_10pts_01.npy','fkpp_1d_early_time_4_10pts_05.npy','fkpp_1d_early_time_4_10pts_10.npy']

# 2D Fisher KPP
#path = '../../Data/Surfaces/FKPP2D/'
#files = [
#    'fkpp_2d_1_03pts_01.npy','fkpp_2d_1_03pts_05.npy','fkpp_2d_2_03pts_10.npy',
#    'fkpp_2d_1_05pts_01.npy','fkpp_2d_1_05pts_05.npy','fkpp_2d_2_05pts_10.npy',
#    'fkpp_2d_1_10pts_01.npy','fkpp_2d_1_10pts_05.npy','fkpp_2d_2_10pts_10.npy',
#    'fkpp_2d_early_time_1_03pts_01.npy','fkpp_2d_early_time_1_03pts_05.npy','fkpp_2d_early_time_1_03pts_10.npy',
#    'fkpp_2d_early_time_1_05pts_01.npy','fkpp_2d_early_time_1_05pts_05.npy','fkpp_2d_early_time_1_05pts_10.npy',
#    'fkpp_2d_early_time_1_10pts_01.npy','fkpp_2d_early_time_1_10pts_05.npy','fkpp_2d_early_time_1_10pts_10.npy',
#    'fkpp_2d_2_03pts_01.npy','fkpp_2d_2_03pts_05.npy','fkpp_2d_2_03pts_10.npy',
#    'fkpp_2d_2_05pts_01.npy','fkpp_2d_2_05pts_05.npy','fkpp_2d_2_05pts_10.npy',
#    'fkpp_2d_2_10pts_01.npy','fkpp_2d_2_10pts_05.npy','fkpp_2d_2_10pts_10.npy',
#    'fkpp_2d_early_time_2_03pts_01.npy','fkpp_2d_early_time_2_03pts_05.npy','fkpp_2d_early_time_2_03pts_10.npy',
#    'fkpp_2d_early_time_2_05pts_01.npy','fkpp_2d_early_time_2_05pts_05.npy','fkpp_2d_early_time_2_05pts_10.npy',
#    'fkpp_2d_early_time_2_10pts_01.npy','fkpp_2d_early_time_2_10pts_05.npy','fkpp_2d_early_time_2_10pts_10.npy',
#    'fkpp_2d_3_03pts_01.npy','fkpp_2d_3_03pts_05.npy','fkpp_2d_3_03pts_10.npy',
#    'fkpp_2d_3_05pts_01.npy','fkpp_2d_3_05pts_05.npy','fkpp_2d_3_05pts_10.npy',
#    'fkpp_2d_3_10pts_01.npy','fkpp_2d_3_10pts_05.npy','fkpp_2d_3_10pts_10.npy',
#    'fkpp_2d_early_time_3_03pts_01.npy','fkpp_2d_early_time_3_03pts_05.npy','fkpp_2d_early_time_3_03pts_10.npy',
#    'fkpp_2d_early_time_3_05pts_01.npy','fkpp_2d_early_time_3_05pts_05.npy','fkpp_2d_early_time_3_05pts_10.npy',
#    'fkpp_2d_early_time_3_10pts_01.npy','fkpp_2d_early_time_3_10pts_05.npy','fkpp_2d_early_time_3_10pts_10.npy',
#    'fkpp_2d_4_03pts_01.npy','fkpp_2d_4_03pts_05.npy','fkpp_2d_4_03pts_10.npy',
#    'fkpp_2d_4_05pts_01.npy','fkpp_2d_4_05pts_05.npy','fkpp_2d_4_05pts_10.npy',
#    'fkpp_2d_4_10pts_01.npy','fkpp_2d_4_10pts_05.npy','fkpp_2d_4_10pts_10.npy',
#    'fkpp_2d_early_time_4_03pts_01.npy','fkpp_2d_early_time_4_03pts_05.npy','fkpp_2d_early_time_4_03pts_10.npy',
#    'fkpp_2d_early_time_4_05pts_01.npy','fkpp_2d_early_time_4_05pts_05.npy','fkpp_2d_early_time_4_05pts_10.npy',
#    'fkpp_2d_early_time_4_10pts_01.npy','fkpp_2d_early_time_4_10pts_05.npy','fkpp_2d_early_time_4_10pts_10.npy']

# load data
file_name = files[options.file_idx]
file = np.load(path + file_name, allow_pickle=True, encoding='latin1').item()
inputs = file['inputs']
shapes = file['shape']
outputs = file['outputs']
gamma = file['gamma']

# 0/1 normalize data
def normalize(x):
    x_min = np.min(x)
    x -= x_min
    x_max = np.max(x)
    x /= x_max
    return x, x_min, x_max

# unpack variables
try: # try 2D
    x = inputs[:, 0].astype(np.float)
    x, x_min, x_max = normalize(x)
    y = inputs[:, 1].astype(np.float)
    y, y_min, y_max = normalize(y)
    t = inputs[:, 2].astype(np.float)
    t, t_min, t_max = normalize(t)
    inputs = np.concatenate([x[:, None], y[:, None], t[:, None]], axis=1)
except: # else 1D
    x = inputs[:, 0].astype(np.float)
    x, x_min, x_max = normalize(x)
    t = inputs[:, 1].astype(np.float)
    t, t_min, t_max = normalize(t)
    inputs = np.concatenate([x[:, None], t[:, None]], axis=1)
u = outputs[:, 0].astype(np.float)
outputs = u[:, np.newaxis]

# get extrema
u_max = np.max(outputs)
u_min = np.min(outputs)

# split train/val 
N = len(inputs)
p = np.random.permutation(N)
split = int(0.8*N)
x_train = inputs[p[:split]]
y_train = outputs[p[:split]]
x_val = inputs[p[split:]]
y_val = outputs[p[split:]]

# conver to torch
def numpy_to_tensor(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr
x_train = numpy_to_tensor(x_train)
y_train = numpy_to_tensor(y_train)
x_val = numpy_to_tensor(x_val)
y_val = numpy_to_tensor(y_val)
inputs = numpy_to_tensor(inputs)
outputs = numpy_to_tensor(outputs)

# loss function
loss_fun = WGLSLoss(gamma=gamma, 
                    threshold=1e-10, 
                    target_weight=1.0, 
                    target_val=None)

# constrain surface to within U_min and U_max
class Regularization(nn.Module):
    
    def __init__(self, num_inputs):
        
        super().__init__()
        self.num_inputs = num_inputs
        if num_inputs == 2:
            x = np.linspace(0, 1, 100)
            t = np.linspace(0, 1, 100)
            X, T = np.meshgrid(x, t, indexing='ij')
            inputs = np.concatenate([X.reshape(-1, 1), 
                                     T.reshape(-1, 1)], axis=1)
            self.inputs = numpy_to_tensor(inputs)
        
    def forward(self, model, inputs, true, pred):
        
        if self.num_inputs == 3:
            x = torch.rand(int(1e4), 1, requires_grad=True) 
            y = torch.rand(int(1e4), 1, requires_grad=True) 
            t = torch.rand(int(1e4), 1, requires_grad=True)
            self.inputs = torch.cat([x, y, t], dim=1).float().to(device)
        
        # predict surface fitter at grid points
        outputs = model(self.inputs)
        
        penalty = torch.mean(torch.where(
            outputs > u_max, (outputs-u_max)**2, torch.zeros_like(outputs)))
        penalty += torch.mean(torch.where(
            outputs < u_min, (outputs-u_min)**2, torch.zeros_like(outputs)))
        
        return penalty

# initialize model wrapper
model = ModelWrapper(
    None,
    None,
    loss_fun,
    regularizer=Regularization(num_inputs=inputs.shape[1]),
    save_name='../Weights/'+path[-7:-1]+'/'+file_name[:-4],
    save_best_train=False,
    save_best_val=True,
    save_opt=False)

# run the model
epochs = int(1e6)
batch_size = 10 if inputs.shape[1] == 2 else 100
min_val_loss = 1e12
for i in range(options.num_reps):
    
    surface_fitter = BuildSurfaceFitter(
        input_variables=inputs.shape[1], 
        hidden_layers=[256, 256, 256], 
        output_variables=outputs.shape[1], 
        activations=[nn.Sigmoid()],
        output_activation=nn.Softplus()) 
    surface_fitter = surface_fitter.to(device)
    opt = torch.optim.Adam(surface_fitter.parameters(), lr=1e-3)
    
    model.model = surface_fitter
    model.optimizer = opt
    
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=options.verbosity,
        validation_data=[x_val, y_val],
        early_stopping=1000,
        best_val_loss=min_val_loss,
        rel_save_thresh=0.01)
    
    if np.min(model.val_loss_list) < min_val_loss:
        min_val_loss = np.min(model.val_loss_list)
