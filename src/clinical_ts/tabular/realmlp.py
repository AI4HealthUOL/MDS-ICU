__all__ = ['RealMLP', 'RealMLPConfig']

from .realmlp_modules.realmlp import RealMLPModule, RealMLPPreprocessing

import dataclasses
from dataclasses import dataclass

from .base import EncoderStaticBase, BasicEncoderStaticConfig

# static_stats_train before target_dim

class RealMLP(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, static_stats_train, target_dim)

        
        if(hparams_encoder_static.preprocessing):
            self.preprocessing = RealMLPPreprocessing(hparams_encoder_static.vocab_sizes, static_stats_train.median, static_stats_train.quantile25, static_stats_train.quantile75, static_stats_train.min, static_stats_train.max)
            input_dim = int(self.preprocessing.get_output_dim())
            print(self.preprocessing.get_output_dim())
            print(int(self.preprocessing.get_output_dim()))
        else:#discouraged: use without preprocessing
            assert(hparams_input_shape.static_dim_cat==0)
            self.preprocessing = None
            input_dim = int(hparams_input_shape.static_dim)

        self.realmlp = RealMLPModule(in_features=input_dim, out_features=hparams_encoder_static.output_dim if target_dim is None else target_dim, hidden_features= hparams_encoder_static.hidden_dim, layers=hparams_encoder_static.layers, activation=hparams_encoder_static.activation)
            
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = int(hparams_encoder_static.output_dim if target_dim is None else target_dim)
        self.output_shape.static_dim_cat = 0
    
    def forward(self, **kwargs):
        xcat = kwargs["static_cat"].long() if "static_cat" in kwargs.keys() else None
        xcont = kwargs["static"] if "static" in kwargs.keys() else None
        return {"static":self.realmlp(self.preprocessing(xcat,xcont)) if self.preprocessing else self.realmlp(xcont)}
    
    def get_output_shape(self):
        return self.output_shape

    def get_param_groups(self, lr):
        return self.realmlp.get_param_groups(lr)

@dataclass
class RealMLPConfig(BasicEncoderStaticConfig):
    _target_:str = "clinical_ts.tabular.realmlp.RealMLP"
    output_dim:int = 64
    hidden_dim:int = 256
    layers:int = 4
    preprocessing:bool = True
    activation:str = "selu" #selu for classification, mish for regression?
    