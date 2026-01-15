#preprocessing adapted from preprocessing.py in realmlp
from .mlp import *
import torch.nn as nn
import numpy as np
import torch

class CustomOneHotEncoder(torch.nn.Module):
    def __init__(self, cat_dims):
        """
        Constructor: Initializes the encoder with the number of categories for each feature.
        cat_dims: List of integers, where each integer represents the number of categories for a feature.
        """
        super(CustomOneHotEncoder, self).__init__()
        self.cat_dims = cat_dims

    def forward(self, X):
        """
        Forward pass: Transforms the input data into one-hot encoded format.
        X: Input data (PyTorch tensor of shape [n_samples, n_features]).
        """
        n_samples, n_features = X.shape
        out_arrs = []

        for i, cat_size in enumerate(self.cat_dims):
            column = X[:, i]  # Feature column
            isnan = torch.isnan(column)  # Mask for NaN values
            column = torch.where(isnan, torch.zeros_like(column), column)  # Replace NaNs with 0 (temporary)
            column = column.long()  # Convert to integer indices

            # Initialize one-hot encoded tensor
            out_arr = torch.zeros((n_samples, cat_size), dtype=torch.float32, device=X.device)

            # One-hot encode the column (ignoring NaNs)
            valid_mask = ~isnan  # Mask for valid (non-NaN) values
            out_arr[valid_mask, column[valid_mask]] = 1.0

            if cat_size == 2:
                # Binary case: encode as -1, 1, or 0 (for missing/unknown values)
                out_arr = out_arr[:, 0:1] - out_arr[:, 1:2]

            out_arrs.append(out_arr)

        # Concatenate all one-hot encoded features along the last dimension
        return torch.cat(out_arrs, dim=-1)


class RobustScaleSmoothClipTransform(nn.Module):
    def __init__(self, median, quantile25, quantile75, min, max):
        super(RobustScaleSmoothClipTransform, self).__init__()
        
        # Ensure X is a numpy array for compatibility with the original code
        #if isinstance(X, torch.Tensor):
        #    X = X.numpy()
        
        # Fit the transformation parameters
        #self._median = torch.tensor(np.median(X, axis=-2), dtype=torch.float32)
        #quant_diff = torch.tensor(np.quantile(X, 0.75, axis=-2), dtype=torch.float32) - torch.tensor(np.quantile(X, 0.25, axis=-2), dtype=torch.float32)
        #max = torch.tensor(np.max(X, axis=-2), dtype=torch.float32)
        #min = torch.tensor(np.min(X, axis=-2), dtype=torch.float32)
        #self._median = torch.tensor(median, dtype=torch.float32)
        self.register_buffer("_median",torch.tensor(median, dtype=torch.float32))
        quant_diff = torch.tensor(quantile75, dtype=torch.float32)-torch.tensor(quantile25, dtype=torch.float32)
        max = torch.tensor(max, dtype=torch.float32)
        min = torch.tensor(min, dtype=torch.float32)

        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        factors[quant_diff == 0.0] = 0.0
        
        self.register_buffer("_factors",factors)
        #self._factors = factors

    def forward(self, X):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / torch.sqrt(1 + (x_scaled / 3) ** 2)


class RealMLPPreprocessing(nn.Module):
    def __init__(self, cat_dims, train_median, train_quantile25, train_quantile75, train_min, train_max):
        super().__init__()
        self.one_hot_enc = CustomOneHotEncoder(cat_dims) if len(cat_dims)>0 else None
        self.scaler = RobustScaleSmoothClipTransform(train_median, train_quantile25, train_quantile75, train_min, train_max) if len(train_median)>0 else None
        self.output_dim = np.sum([(1 if c==2 else c) for c in cat_dims])+len(train_median)
        
    def forward(self, x_cat, x_cont):
        if(self.scaler is not None):
            x_cont= self.scaler(x_cont)
        if(self.one_hot_enc):
            x_cat = self.one_hot_enc(x_cat)
        if(x_cat is None):
            return x_cont
        elif(x_cont is None):
            return x_cat
        else:
            return torch.cat([x_cat,x_cont],axis=1)

    def get_output_dim(self):
        return self.output_dim
        

class RealMLPModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features:int = 256, layers: int=4, activation:str="selu"):
        super().__init__()
        
        if(activation=="selu"):# classification
            act = nn.SELU
        elif(activation=="mish"):#regression
            act = Mish
        else:
            assert(False)
  
        modules = [ScalingLayer(in_features)]
        for i in range(layers-1):
            modules = modules + [NTPLinear(in_features if i==0 else hidden_features, hidden_features), act()]
        modules.append(NTPLinear(hidden_features, out_features, zero_init=True))
        self.model = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.model(x)
        
    def get_param_groups(self, lr):
        params = list(self.model.parameters())
        scale_params = self.model[0].parameters()
        weights = [x.weight for x in self.model if isinstance(x,NTPLinear)]
        biases = [x.bias for x in self.model if isinstance(x,NTPLinear)]
        return [{"params": scale_params, "lr":6*lr},{"params": weights, "lr":lr},{"params": biases, "lr":0.1*lr}]