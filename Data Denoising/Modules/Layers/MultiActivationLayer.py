import torch, pdb
import torch.nn as nn

class MultiActivationLayer(nn.Module):
    
    '''
    Constructs hidden layer with multiple activation functions.
    For each activation function there is a linear transformation
    followed by the corresponding nonlinear activation. If aggregate 
    is set to true, the individual dense layers are processed by an 
    additional linear transformation to bring the dimensionality 
    back down to the specified number of neurons, otherwise the 
    number of outputs will be output_features * len(activations). 
    
    Args:
        input_features:  integer number of input features
        output_features: integer number of output features
        activations:     list of instantiated activation functions
        aggregate:       boolean indicator for combining outputs
    
    Inputs:
        x: torch float tensor of inputs
    
    Returns:
        x: torch float tensor of outputs
    '''
    
    def __init__(self, 
                 input_features, 
                 output_features, 
                 activations=None, 
                 aggregate=True):
        
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.aggregate = aggregate
        
        # list of nonlinear activations
        activations = activations if activations is not None else nn.Softplus()
        if not isinstance(activations, list):
            self.activations = [activations] 
        else:
            self.activations = activations
        
        # big linear activation
        self.linear_activation = nn.Linear(
            in_features=self.input_features,
            out_features=len(self.activations)*self.output_features,
            bias=True)
        
        # optional linear aggregation
        if len(self.activations) == 1:
            self.aggregate = False
        if self.aggregate:
            self.output_activation = nn.Linear(
                in_features=len(self.activations)*self.output_features,
                out_features=self.output_features,
                bias=True)
        
    def forward(self, x):
        
        # linear activation
        x = self.linear_activation(x)
            
        # slice neurons for each activation function
        n = self.output_features # number of neurons per activation
        a = len(self.activations) # number of activations
        x = [x[:, j*n:(j+1)*n] for j in range(a)]
            
        # apply activations to slices
        x = [self.activations[j](x[j]) for j in range(a)]
        
        # combine
        x = torch.cat(x, dim=1)
        
        # optional linear aggregation
        if self.aggregate:
            x = self.output_activation(x)
        
        return x
        
        