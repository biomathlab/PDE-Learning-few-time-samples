import torch, pdb
import torch.nn as nn

from Modules.Activations.SoftplusReLU import SoftplusReLU
from Modules.Layers.MultiActivationLayer import MultiActivationLayer

class BuildSurfaceFitter(nn.Module):
   
    '''
    Builds a custom multilayer perceptron (MLP) for surface fitting. This
    class allows the user to specify multiple activation functions per
    layer. 
   
    Args:
        input_variables:   integer number of input variables
        hidden_layers:     list of integer hidden layer sizes
        output_variables:  integer number of output variables
        activations:       list of instantiated activation functions
        output_activation: instantiated activation function
   
    Inputs:
        x: torch float tensor of input variables
   
    Returns:
        y: torch float tensor of output variables
    '''
   
    def __init__(self, 
                 input_variables,
                 hidden_layers,
                 output_variables,
                 activations=None,
                 output_activation=None):
        
        super().__init__()
        self.input_variables = input_variables
        self.hidden_layers = hidden_layers
        self.output_variables = output_variables
        self.output_activation = output_activation
            
        # list of nonlinear activations
        activations = activations if activations is not None else SoftplusReLU()
        if not isinstance(activations, list):
            self.activations = [activations] 
        else:
            self.activations = activations
            
        # aggregate if using multiple activations
        if len(self.activations) > 1:
            self.aggregate = True
        else:
            self.aggregate = False
        
        # build surface fitter
        layers = []
        for i, layer in enumerate(self.hidden_layers):
            layers.append(MultiActivationLayer(
                input_features=self.input_variables, 
                output_features=layer, 
                activations=self.activations, 
                aggregate=self.aggregate))
            self.input_variables = layer
        
        # linear output layer
        self.output_layer = nn.Linear(
            in_features=self.input_variables,
            out_features=output_variables,
            bias=True)
        
        # nonlinear output
        if self.output_activation is not None:
            self.output_layer = nn.Sequential(
                self.output_layer,
                self.output_activation)
            
        # combine into sequential model
        layers.append(self.output_layer)
        self.surface_fitter = nn.Sequential(*layers)
       
    def forward(self, x):
       
        # run the model
        y = self.surface_fitter(x)
       
        return y
