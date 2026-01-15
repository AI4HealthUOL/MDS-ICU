from ..template_model import SSLModel
from ..template_modules import TaskConfig
from ..data.time_series_dataset_utils import load_dataset
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

#disable performance warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class MultimodalModel(SSLModel):
    '''class for multimodal tasks'''
   
    def preprocess_dataset(self, dataset_kwargs):

        df_mapped, lbl_itos, mean, std = load_dataset(Path(dataset_kwargs.path))

        if self.hparams.loss.loss_type == "supervised" and dataset_kwargs.name.startswith("mdsicu"):

            # Load data
            print('loading data')
            df_features = pd.read_parquet(Path(dataset_kwargs.path) / "mds_icu_preprocessed2_ecg.parquet")
            print(len(df_features['features'].iloc[0]))
            print(len(df_features['label'].iloc[0]))
            print('data loaded')

        return df_features, lbl_itos, mean, std



    
@dataclass
class TaskConfigMultimodal(TaskConfig):
    mainclass:str= "clinical_ts.task.multimodal.MultimodalModel"
    introduce_nan_columns:bool = False #impute using train set median and introduce an additional column that states if imputation occurred 
    nan_columns_as_cat:bool = False #treat nan columns as categorical variables (instead of continuous)
