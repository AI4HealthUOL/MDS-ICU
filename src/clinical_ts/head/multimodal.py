__all__ = ['ConcatFusionHead', 'ConcatFusionHeadConfig','TensorFusionHead', 'TensorFusionHeadConfig', 'AttentionFusionHead', 'AttentionFusionHeadConfig', 'PoolingConcatFusionHeadConfig']

import torch
import torch.nn as nn

from ..template_modules import HeadBase, HeadBaseConfig
from ..ts.transformer_modules.transformer import TransformerHead
import dataclasses
from dataclasses import dataclass, field

from ..ts.head import SequentialHeadConfig, PoolingHeadConfig


class ConcatFusionHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''Simple concatenation plus linear head'''
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        input_size = (hparams_input_shape.length*hparams_input_shape.channels if hparams_input_shape.length>0 else hparams_input_shape.channels)+hparams_input_shape.static_dim+hparams_input_shape.static_dim_cat
        self.linear = nn.Linear(input_size,target_dim)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = 0
        self.output_shape.static_dim = 0
        self.output_shape.static_dim_cat = 0
        
    def forward(self, **kwargs):
        static = kwargs["static"]
        seq = kwargs["seq"]
        return {"seq": self.linear(torch.cat((seq.view(seq.shape[0],-1),static),dim=1))}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class ConcatFusionHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.head.multimodal.ConcatFusionHead"
    multi_prediction:bool=False

class TensorFusionHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''Tensor fusion head from https://arxiv.org/abs/1707.07250'''
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        input_size = (1+hparams_input_shape.length*hparams_input_shape.channels if hparams_input_shape.length>0 else 1+hparams_input_shape.channels)*(1+hparams_input_shape.static_dim+hparams_input_shape.static_dim_cat)
        self.linear = nn.Linear(input_size,target_dim)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = 0
        self.output_shape.static_dim = 0
        self.output_shape.static_dim_cat = 0
        
    def forward(self,**kwargs):
        seq = kwargs["seq"]
        x = seq.view(seq.shape[0],-1)
        y = kwargs["static"]
        x1 = torch.cat((x,torch.ones(x.shape[0],1,device=x.device)),dim=1)
        y1 = torch.cat((y,torch.ones(x.shape[0],1,device=y.device)),dim=1)
        x1y1 = (x1.unsqueeze(dim=1)*y1.unsqueeze(dim=2)).view(x.shape[0],-1)
        return {"seq": self.linear(x1y1)}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class TensorFusionHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.head.multimodal.TensorFusionHead"
    multi_prediction:bool=False

class AttentionFusionHead(HeadBase):
    '''concatenation plus generalization of seq_pool from Compact-Transformers (multi-head attention)'''
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)
        assert(hparams_head.multi_prediction is False)
        #assert(hparams.predictor._target_!="clinical_ts.ts.transformer.TransformerPredictor" or (hparams.predictor.cls_token is True or (hparams.predictor.cls_token is False and (hparams_head.head_pooling_type!="cls" and hparams_head.head_pooling_type!="meanmax-cls"))))

        self.head = TransformerHead(1,target_dim,pooling_type="seq",batch_first=True,n_heads_seq_pool=hparams_head.head_n_heads_seq_pool)

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = 0
        self.output_shape.static_dim = 0
        self.output_shape.static_dim_cat = 0

    def forward(self,**kwargs):
        seq = kwargs["seq"]
        x = torch.cat((seq.view(seq.shape[0],-1),kwargs["static"]),dim=1)
        return {"seq": self.head(x.unsqueeze(dim=1))}#expects bs,seq,feat
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class AttentionFusionHeadConfig(HeadBaseConfig):
    _target_ = "clinical_ts.head.multimodal.AttentionFusionHead"
    multi_prediction:bool=False
    head_n_heads_seq_pool:int=10

@dataclass
class PoolingConcatFusionHeadConfig(SequentialHeadConfig):
    head0: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)#typically a global pooling
    head1: ConcatFusionHeadConfig = field(default_factory=ConcatFusionHeadConfig)
