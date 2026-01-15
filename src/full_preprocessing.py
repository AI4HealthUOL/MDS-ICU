import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
from datetime import timedelta
from tqdm import tqdm
import glob
from sklearn.linear_model import LinearRegression
from collections import Counter
import os
import json
from natsort import natsorted

ecg_utils import prepare_mimicecg
timeseries_utils import reformat_as_memmap

import argparse
from pathlib import Path
import icd10

from tqdm import tqdm
!pip -q install dtype_diet
from dtype_diet import report_on_dataframe, optimize_dtypes

from scipy.stats import iqr

pd.options.mode.chained_assignment = None


# load
zip_file_path = Path('data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip') 
target_path = Path('data/memmap/') 
df,_,_,_=prepare_mimicecg(zip_file_path, target_folder=target_path)


reformat_as_memmap(df, 
                   target_path/"memmap.npy", 
                   data_folder=target_path, 
                   annotation=False, 
                   max_len=0, 
                   delete_npys=True, 
                   col_data="data", 
                   col_lbl=None, 
                   batch_length=0, 
                   skip_export_signals=False)        

# load
df_diags = pd.read_csv('data/records_w_diag_icd10.csv')
df_diags['all_diag_all']=df_diags['all_diag_all'].apply(lambda x:eval(x))

df_diags = df_diags[df_diags['all_diag_all'].apply(lambda x: len(x) > 0)]
df_diags["all_diag_all"]=df_diags["all_diag_all"].apply(lambda x: list(set([y.strip()[:5] for y in x])))
df_diags["all_diag_all"]=df_diags["all_diag_all"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
df_diags['all_diag_all'] = df_diags['all_diag_all'].apply(lambda code: code[:-1] if code[-1].isalpha() else code)

def label_propagation(df, column_name, propagate_all=True):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    col_flat = flatten(np.array(df[column_name]))
    
    def prepare_consistency_mapping_internal(codes_unique, codes_unique_all):
        res={}
        for c in codes_unique:
            if(propagate_all):
                res[c]=[c[:i] for i in range(3,len(c)+1)]
            else: 
                res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
        return res
    
    cons_map = prepare_consistency_mapping_internal(np.unique(col_flat), np.unique(col_flat))
    df[column_name] = df[column_name].apply(lambda x: list(set(flatten([cons_map[y] for y in x]))))
    
    return df

df_diags = label_propagation(df_diags, "all_diag_all", propagate_all=True)

def process_multilabel(df, column_name, threshold, output_column_name, fold_column, fold_18, fold_19):

    df_fold_18 = df[df[fold_column] == fold_18]
    df_fold_19 = df[df[fold_column] == fold_19]
    df_remaining = df[~df[fold_column].isin([fold_18, fold_19])]  
    
    counts_fold_18 = {}
    for row in df_fold_18[column_name]:
        for item in row:
            counts_fold_18[item] = counts_fold_18.get(item, 0) + 1
    
    counts_fold_19 = {}
    for row in df_fold_19[column_name]:
        for item in row:
            counts_fold_19[item] = counts_fold_19.get(item, 0) + 1
    
    counts_remaining = {}
    for row in df_remaining[column_name]:
        for item in row:
            counts_remaining[item] = counts_remaining.get(item, 0) + 1

    valid_labels = {
        item for item in set(counts_fold_18.keys()) & set(counts_fold_19.keys()) & set(counts_remaining.keys())
        if counts_fold_18.get(item, 0) >= threshold
        and counts_fold_19.get(item, 0) >= threshold
        and counts_remaining.get(item, 0) >= threshold
    }

    total_counts = {item: counts_fold_18.get(item, 0) + counts_fold_19.get(item, 0) + counts_remaining.get(item, 0)
                    for item in valid_labels}
    
    unique_strings = sorted(total_counts, key=total_counts.get, reverse=True)

    df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in unique_strings])

    return df, np.array(unique_strings)


df_diags, lbls_diags = process_multilabel(df_diags, 'all_diag_all', 10, 'Diagnoses_labels', 'strat_fold', 18, 19)
np.save('lbl_diags.npy', lbls_diags)

# load
df_icustays = pd.read_csv('data/icustays.csv.gz') 

df_diags = df_diags[df_diags['hosp_hadm_id'].isin(df_icustays['hadm_id'].unique())]

df_diags['ecg_time'] = pd.to_datetime(df_diags['ecg_time'])
df_diags['dod'] = pd.to_datetime(df_diags['dod'])
df_icustays['icu_intime'] = pd.to_datetime(df_icustays['intime'])
df_icustays['icu_outtime'] = pd.to_datetime(df_icustays['outtime'])

# load
df_hosp = pd.read_csv('data/admissions.csv.gz')
df_hosp['hosp_dischtime'] = pd.to_datetime(df_hosp['dischtime'])

df_icustays = df_icustays.merge(df_hosp[['hadm_id', 'hosp_dischtime', 'race']], on='hadm_id', how='left')

rows_to_keep = []

for _, row in tqdm(df_diags.iterrows()):
    patient = row['subject_id']
    ecg_time = row['ecg_time']

    df_patient = df_icustays[df_icustays['subject_id'] == patient]
  
    for _, patient_row in df_patient.iterrows(): 
        intime = patient_row['icu_intime']
        outtime = patient_row['icu_outtime']

        if intime <= ecg_time and ecg_time <= outtime:
            row_dict = row.to_dict()
            row_dict['icu_stay_id'] = patient_row['stay_id']
            row_dict['icu_intime'] = patient_row['icu_intime']
            row_dict['icu_outtime'] = patient_row['icu_outtime']
            row_dict['hosp_dischtime'] = patient_row['hosp_dischtime']
            row_dict['race'] = patient_row['race']
            
            rows_to_keep.append(row_dict)

df_filtered_diags = pd.DataFrame(rows_to_keep)

def map_race(row):
    if 'WHITE' in row:
        return 0
    elif 'BLACK' in row:
        return 1
    elif 'HISPANIC' in row:
        return 2
    elif 'ASIAN' in row:
        return 3
    else:
        return 4


df_filtered_diags['race_mapped'] = df_filtered_diags['race'].apply(map_race)
df_filtered_diags['gender'] = df_filtered_diags['gender'].apply(lambda x: 1 if x=='M' else 0)

def augment_context_windows(df, global_seed=42):
    
    augmented_rows = []

    for idx, row in tqdm(df.iterrows()):
        icu_start = row['icu_intime']
        icu_end = row['icu_outtime']
        ecg_time = row['ecg_time']
        base_row = row.to_dict()

        context_start = icu_start
        context_end = ecg_time
        context_len = context_end - context_start
        augmented_rows.append({**base_row, 'context_start': context_start, 'context_end': context_end,
                               'context_len': context_len, 'augmented': 0})

        unique_seed = hash((row['subject_id'], global_seed)) % (2**32)
        rng = np.random.default_rng(unique_seed)
        
        window_len = ecg_time - icu_start  

        ecg_pos_ranges = {
            1: (0.0, 0.25),   
            2: (0.25, 0.5),  
            3: (0.5, 0.75),   
            4: (0.75, 1.0)   
        }
        

        coverage_ranges = {
            1: (0.0, 0.25),
            2: (0.25, 0.5),
            3: (0.5, 0.75),
            4: (0.75, 1.0)
        }

        
        n_aug_per_combo = 1  
        icu_duration = icu_end - icu_start  
        icu_duration_sec = icu_duration.total_seconds()


        for ecg_type, (start_frac, end_frac) in ecg_pos_ranges.items():
            for coverage_type, (cov_start, cov_end) in coverage_ranges.items():
                for i in range(n_aug_per_combo):
                    coverage_ratio = rng.uniform(cov_start, cov_end)
                    window_len = pd.Timedelta(seconds=coverage_ratio * icu_duration_sec)

                    rel_pos = rng.uniform(start_frac, end_frac)
                    ecg_offset = pd.Timedelta(seconds=rel_pos * window_len.total_seconds())

                    context_start = ecg_time - ecg_offset
                    context_end = context_start + window_len

                    if context_start >= icu_start and context_end <= icu_end:
                        context_len = context_end - context_start
                        augmented_id = int(f"{ecg_type}{coverage_type}{i}")

                        augmented_rows.append({
                            **base_row,
                            'context_start': context_start,
                            'context_end': context_end,
                            'context_len': context_len,
                            'augmented': augmented_id
                        })




    return pd.DataFrame(augmented_rows)


df_filtered_diags = augment_context_windows(df_filtered_diags)

df_filtered_diags = df_filtered_diags[
    (df_filtered_diags['ecg_time'] >= df_filtered_diags['context_start']) &
    (df_filtered_diags['ecg_time'] <= df_filtered_diags['context_end'])
]

df_filtered_diags = df_filtered_diags[~df_filtered_diags.duplicated(subset=['subject_id', 'context_start', 'context_end'])]
df_filtered_diags["ecg_position_ratio"] = (
    (df_filtered_diags["ecg_time"] - df_filtered_diags["context_start"])
    / (df_filtered_diags["context_end"] - df_filtered_diags["context_start"])
)

df_filtered_diags['coverage_ICU'] = (
    (df_filtered_diags['context_end'] - df_filtered_diags['context_start']) /
    (df_filtered_diags['icu_outtime'] - df_filtered_diags['icu_intime'])
)

df_filtered_diags['context_len_hours'] = df_filtered_diags['context_len'].dt.total_seconds() / 3600

min_window_hours = 1

df_filtered_diags = df_filtered_diags[
    df_filtered_diags['context_len_hours'] >= min_window_hours
]

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
# load
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_filtered_diags['subject_id'].unique())
relevant_stay_ids = set(df_filtered_diags['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)
    
filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]
label_counts = filtered_chartevents_df['label'].value_counts()
labels_to_keep = label_counts[label_counts >= 1000].index
filtered_chartevents_df = filtered_chartevents_df[filtered_chartevents_df['label'].isin(labels_to_keep)]

base_terms = ['Albumin',
              'Alkaline Phosphate',
              'Differential-Basos',
              'Absolute Count - Basos',
              'Arterial Base Excess',
              'Total Bilirubin',
              'Direct Bilirubin',
              'Ionized Calcium',
              'Calcium non-ionized',
              'Chloride (serum)',
              'Chloride (whole blood)',
              'Creatinine (serum)',
              'C Reactive Protein (CRP)',
              'Differential-Eos',
              'Absolute Count - Eos', 
              'Fibrinogen',
              'Glucose (serum)',
              'Glucose finger stick (range 70-100)',
              'Glucose (whole blood)',
              'Hematocrit (serum)',
              'Hematocrit (whole blood - calc)',
              'Hemoglobin',
              'Potassium (serum)',
              'Potassium (whole blood)',
              'INR',
              'Differential-Lymphs',
              'Absolute Count - Lymphs',
              'Hemoglobin',
              'Magnesium',
              'Sodium (serum)',
              'Sodium (whole blood)',
              'Differential-Neuts',
              'Absolute Count - Neuts',
              'PAR-Oxygen saturation',
              'Arterial CO2 Pressure',
              'EtCO2',
              'PeCO2',
              'PTT',
              'Platelet Count',
              'Arterial O2 pressure',
              'Prothrombin time',
              'CO2 production',
              'Troponin-T',
              'WBC']

df_icu_labvalues = filtered_chartevents_df[filtered_chartevents_df['label'].isin(base_terms)]
df_icu_labvalues = df_icu_labvalues[df_icu_labvalues['subject_id'].isin(df_filtered_diags['subject_id'].unique())]

absolute_threshold = 0.8e6

cleaned_list = []

for label in df_icu_labvalues['label'].unique():
    df_label = df_icu_labvalues[df_icu_labvalues['label'] == label].dropna(subset=['valuenum'])
    df_label_cleaned = df_label[df_label['valuenum'] < absolute_threshold]
    cleaned_list.append(df_label_cleaned[['subject_id','charttime','storetime','itemid','valuenum','valueuom','label']])

df_icu_labvalues = pd.concat(cleaned_list, ignore_index=True)

# load
df_labtitles = pd.read_csv('data/d_labitems.csv.gz', compression='gzip')
# load
df_labevents = pd.read_csv('data/labevents.csv.gz', compression='gzip', low_memory=False)

df_labs = pd.merge(df_labevents, df_labtitles, on='itemid')
df_labevents = None
df_labs = df_labs[df_labs['subject_id'].isin(df_filtered_diags['subject_id'].unique())]

base_terms_2 = ['Albumin',
                'Alkaline Phosphatase',
                'Leukocyte Alkaline Phosphatase',
                'Alanine Aminotransferase (ALT)',
               'Basophils','Absolute Basophil Count',
                'Base Excess',
                'Bicarbonate',
               'Bilirubin, Indirect',
                'Bilirubin, Direct',
                'Bilirubin, Total',
                'Urea Nitrogen',
                'Calcium, Total',
               'Free Calcium',
                'Creatine Kinase, MB Isoenzyme',
                'Creatine Kinase (CK)',
                'Chloride',
                'Creatinine',
                'C-Reactive Protein',
               'Eosinophils',
                'Absolute Eosinophil Count',
                'Sedimentation Rate',
                'Fibrinogen, Functional',
                'Glucose',
                'Carboxyhemoglobin',
               'Hematocrit',
                'Hemoglobin',
                'Thrombin',
                'Potassium',
                'Lactate',
                'Lymphocytes',
                'Absolute Lymphocyte Count',
                'Methemoglobin',
               'Magnesium',
                'Sodium',
                'Neutrophils',
                'Absolute Neutrophil Count',
                'Oxygen Saturation',
                'Total CO2',
                'pCO2',
                'pH',
               'Phosphate',
                'Platelet Count',
                'pO2',
                'Red Blood Cells',
                'Troponin T',
                'Troponin I',
                'White Blood Cells',
                'WBC Count']

df_labs = df_labs[df_labs['label'].isin(base_terms_2)]

absolute_threshold = 0.8e6

cleaned_labs_list = []

for label in base_terms_2:
    df_label = df_labs[df_labs['label'] == label]

    df_label_cleaned = df_label[df_label['valuenum'].notna()]

    quantile_threshold = df_label_cleaned['valuenum'].quantile(0.999)

    df_label_cleaned = df_label_cleaned[
        (df_label_cleaned['valuenum'] < absolute_threshold) &
        (df_label_cleaned['valuenum'] <= quantile_threshold)
    ]

    cleaned_labs_list.append(df_label_cleaned[['subject_id', 'charttime', 'storetime', 'itemid', 'valuenum', 'valueuom', 'label']])

df_labs = pd.concat(cleaned_labs_list, ignore_index=True)

df_labs = df_labs[~((df_labs['label'] == 'Absolute Lymphocyte Count') & (df_labs['valueuom'] == '#/uL'))]
df_labs = df_labs[~((df_labs['label'] == 'Creatine Kinase (CK)') & (df_labs['valuenum'] > 25000))]

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Chloride (serum)'], 'Chloride')
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Chloride (whole blood)'))]

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Creatinine (serum)'], 'Creatinine')

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Sodium (serum)'], 'Sodium')
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Sodium (whole blood)'))]

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Hematocrit (serum)'], 'Hematocrit')
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Hematocrit (whole blood - calc)'))]

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Glucose (serum)'], 'Glucose')
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Glucose finger stick (range 70-100)'))]
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Glucose (whole blood)'))]

df_icu_labvalues['label'] = df_icu_labvalues['label'].replace(['Potassium (serum)'], 'Potassium')
df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'Potassium (whole blood)'))]

df_icu_labvalues = df_icu_labvalues[~((df_icu_labvalues['label'] == 'PAR-Oxygen saturation'))]

df_labs = df_labs[~((df_labs['label'] == 'Total CO2'))]

df_labs['label'] = df_labs['label'].replace(['Fibrinogen, Functional'], 'Fibrinogen')

to_use = {
    'Alkaline Phosphatase': 'Alkaline Phosphate',
    'Absolute Basophil Count': 'Absolute Count - Basos',
    'Base Excess': 'Arterial Base Excess',
    'Bilirubin, Direct': 'Direct Bilirubin',
    'Bilirubin, Total': 'Total Bilirubin',
    'Calcium, Total': 'Calcium non-ionized',
    'Free Calcium': 'Ionized Calcium',
    'C-Reactive Protein': 'C Reactive Protein (CRP)',
    'Troponin T': 'Troponin-T',
    'White Blood Cells': 'WBC',
    'pO2': 'Arterial O2 pressure',
    'pCO2': 'Arterial CO2 Pressure'
}

df_labs['label'] = df_labs['label'].map(to_use).fillna(df_labs['label'])

to_use = {
    'Differential-Neuts': 'Neutrophils',
    'Differential-Lymphs': 'Lymphocytes',
    'Differential-Eos': 'Eosinophils',
    'Differential-Basos': 'Basophils',
    'Absolute Count - Eos':'Absolute Eosinophil Count',
    'Absolute Count - Lymphs':'Absolute Lymphocyte Count',
    'Absolute Count - Neuts':'Absolute Neutrophil Count',

}

df_icu_labvalues['label'] = df_icu_labvalues['label'].map(to_use).fillna(df_icu_labvalues['label'])

df_labs['source'] = 'df_labs'
df_icu_labvalues['source'] = 'df_icu_labvalues'
df_combined = pd.concat([df_labs, df_icu_labvalues], ignore_index=True)
df_combined = df_combined[df_combined['label'].map(df_combined['label'].value_counts()) > 1000]
df_combined = df_combined[
    ~((df_combined['label'] == 'Absolute Eosinophil Count') & df_combined['valueuom'].str.contains('%', na=False))
]

df_combined = df_combined[
    ~((df_combined['label'] == 'Absolute Lymphocyte Count') & df_combined['valueuom'].str.contains('%', na=False))
]

df_combined = df_combined[
    ~((df_combined['label'] == 'Absolute Neutrophil Count') & df_combined['valueuom'].str.contains('%', na=False))
]

df_combined = df_combined.drop_duplicates(subset=['subject_id','charttime','valuenum', 'label'])

output_dir_raw = 'preprocessed_labvalues/raw_data/'
os.makedirs(output_dir_raw, exist_ok=True)

labs_grouped = df_combined.groupby('subject_id')

for subject_id, df_labs_subject in tqdm(labs_grouped, total=len(labs_grouped)):
    df_labs_subject[['label', 'valuenum', 'charttime', 'storetime']].to_csv(f'{output_dir_raw}{subject_id}.csv', index=False)
    

output_dir_raw = 'preprocessed_labvalues/raw_data/'

unique_labels = set()

for filename in tqdm(os.listdir(output_dir_raw)):
    if filename.endswith('.csv'):
        file_path = os.path.join(output_dir_raw, filename)
        df = pd.read_csv(file_path, usecols=['label'])
        unique_labels.update(df['label'].dropna().unique())

unique_labels = sorted(unique_labels)

def compute_stats(values, times_in_hours=None, context_end=None, include_hours_before_end=False):
    stats = {}

    if values is None or len(values) == 0:
        base = {
            'min': np.nan,
            'max': np.nan,
            'first': np.nan,
            'last': np.nan
        }
        stats.update(base)

        if include_hours_before_end:
            stats.update({
                'hours_after_min': np.nan,
                'hours_after_max': np.nan,
                'hours_after_first': np.nan,
                'hours_after_last': np.nan
            })
        return stats

    values = np.array(values, dtype=float)

    stats['min'] = np.min(values)
    stats['max'] = np.max(values)
    stats['first'] = values[0]
    stats['last'] = values[-1]

    if not include_hours_before_end:
        return stats

    if (
        times_in_hours is None or 
        len(times_in_hours) != len(values) or 
        context_end is None
    ):
        stats.update({
            'hours_after_min': np.nan,
            'hours_after_max': np.nan,
            'hours_after_first': np.nan,
            'hours_after_last': np.nan
        })
        return stats

    try:
        times = np.array(times_in_hours, dtype=float)
    except Exception:
        stats.update({
            'hours_after_min': np.nan,
            'hours_after_max': np.nan,
            'hours_after_first': np.nan,
            'hours_after_last': np.nan
        })
        return stats

 
    idx_min = np.argmin(values)
    idx_max = np.argmax(values)
    idx_first = 0
    idx_last = len(values) - 1


    try:
        stats['hours_after_min']   = context_end - times[idx_min]
        stats['hours_after_max']   = context_end - times[idx_max]
        stats['hours_after_first'] = context_end - times[idx_first]
        stats['hours_after_last']  = context_end - times[idx_last]
    except Exception:
        stats.update({
            'hours_after_min': np.nan,
            'hours_after_max': np.nan,
            'hours_after_first': np.nan,
            'hours_after_last': np.nan
        })

    return stats


def compute_diffs(values, relative=True):
    values = np.array(values, dtype=float)
    if len(values) <= 1:
        return []

    if relative:
        denom = values[:-1].copy()
        denom[denom == 0] = np.nan  
        diffs = (values[1:] - values[:-1]) / denom
    else:
        diffs = np.diff(values)

    return diffs

input_dir_raw = 'preprocessed_labvalues/raw_data/'
all_labels = set()

csv_files = [f for f in os.listdir(input_dir_raw) if f.endswith('.csv')]

for csv_file in tqdm(csv_files, total=len(csv_files)):
    df = pd.read_csv(os.path.join(input_dir_raw, csv_file), usecols=['label'])
    all_labels.update(df['label'].dropna().unique())

unique_labvalues = np.array(sorted(all_labels))

input_dir_raw = 'preprocessed_labvalues/raw_data/'
output_dir_stats = 'preprocessed_labvalues/stats_data2/'
os.makedirs(output_dir_stats, exist_ok=True)

df_filtered_diags = df_filtered_diags.reset_index(drop=True)

output_dir_stats = 'preprocessed_labvalues/stats_data2/'

batch_files = [f for f in os.listdir(output_dir_stats) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)  

batch_size = 5000

if batch_files:
    last_file = int(batch_files[-1].split('_')[-2])
    file_count = last_file + 1
    start = file_count * batch_size
else:
    file_count = 0
    start = 0
    

buffer = []

raw_keys = [
    'min', 'max', 'first', 'last',
    'hours_after_min', 'hours_after_max',
    'hours_after_first', 'hours_after_last'
]

diff_keys = [
    'min', 'max', 'first', 'last',
    'hours_after_min', 'hours_after_max',
    'hours_after_first', 'hours_after_last'
]

for index, row in tqdm(df_filtered_diags.iloc[start:].iterrows(),
                       total=len(df_filtered_diags),
                       initial=start):

    subject_id = row['subject_id']
    context_start = pd.to_datetime(row['context_start'])
    context_end = pd.to_datetime(row['context_end'])

    raw_data_path = f'{input_dir_raw}{subject_id}.csv'

    row_features = {}  

    if os.path.exists(raw_data_path):
        df_labs_subject = pd.read_csv(raw_data_path)

        df_labs_subject['charttime'] = pd.to_datetime(
            df_labs_subject['charttime'], format="mixed", errors="coerce"
        )
        df_labs_subject['storetime'] = pd.to_datetime(
            df_labs_subject['storetime'], format="mixed", errors="coerce"
        )

        filtered_labs = df_labs_subject[
            (df_labs_subject['charttime'] >= context_start) &
            (df_labs_subject['charttime'] <= context_end) &
            (df_labs_subject['storetime'] >= context_start) &
            (df_labs_subject['storetime'] <= context_end)
        ]


        for labvalue in unique_labvalues:
            labvalue_clean = (
                labvalue.lower()
                        .replace('-', '_').replace(' ', '_')
                        .replace('___', '_').replace('__', '_')
            )

            subset = filtered_labs[filtered_labs['label'] == labvalue]
            subset = subset.sort_values('storetime')

 
            if not subset.empty:
                values = subset['valuenum'].dropna().values
                times_in_hours = (subset['storetime'] - context_start).dt.total_seconds() / 3600

               
                raw_stats = compute_stats(
                    values,
                    times_in_hours=times_in_hours,
                    context_end=context_end.timestamp() / 3600,
                    include_hours_before_end=True
                )

               
                diffs = compute_diffs(values)
                if len(times_in_hours) > 1:
                    diffs_times = times_in_hours[1:]
                else:
                    diffs_times = None

                diff_stats = compute_stats(
                    diffs,
                    times_in_hours=diffs_times,
                    context_end=context_end.timestamp() / 3600,
                    include_hours_before_end=True
                )

               
                for k, v in raw_stats.items():
                    row_features[f'labvalues_raw_{labvalue_clean}_{k}'] = v

                for k, v in diff_stats.items():
                    row_features[f'labvalues_diffs_{labvalue_clean}_{k}'] = v


            else:
                for k in raw_keys:
                    row_features[f'labvalues_raw_{labvalue_clean}_{k}'] = np.nan
                for k in diff_keys:
                    row_features[f'labvalues_diffs_{labvalue_clean}_{k}'] = np.nan


    else:
        for labvalue in unique_labvalues:
            labvalue_clean = (
                labvalue.lower()
                        .replace('-', '_').replace(' ', '_')
                        .replace('___', '_').replace('__', '_')
            )

            for k in raw_keys:
                row_features[f'labvalues_raw_{labvalue_clean}_{k}'] = np.nan
            for k in diff_keys:
                row_features[f'labvalues_diffs_{labvalue_clean}_{k}'] = np.nan

    buffer.append(row_features)

    if len(buffer) >= batch_size:
        batch_df = pd.DataFrame(buffer)
        batch_df.to_csv(f'{output_dir_stats}batch_{file_count}_stats.csv', index=False)
        buffer = []
        file_count += 1

if buffer:
    batch_df = pd.DataFrame(buffer)
    batch_df.to_csv(f'{output_dir_stats}batch_{file_count}_stats.csv', index=False)
    

output_dir_stats = 'preprocessed_labvalues/stats_data2/'

batch_files = [f for f in os.listdir(output_dir_stats) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)  

all_batches = []

for file in tqdm(batch_files, desc="Loading batch files"):
    batch_df = pd.read_csv(os.path.join(output_dir_stats, file))
    all_batches.append(batch_df)

combined_stats_df = pd.concat(all_batches, ignore_index=True)

df_features = pd.concat(
    [df_filtered_diags.reset_index(drop=True), combined_stats_df.reset_index(drop=True)], 
    axis=1
)

df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)


chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
 
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)
    
filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]
label_counts = filtered_chartevents_df['label'].value_counts()
labels_to_keep = label_counts[label_counts >= 1000].index

filtered_chartevents_df = filtered_chartevents_df[filtered_chartevents_df['label'].isin(labels_to_keep)]

category_maps = {
    'Level of Consciousness': {
        'Alert': 0,
        'Awake/Unresponsive': 1,
        'Arouse to Voice': 2,
        'Arouse to Stimulation': 3,
        'Arouse to Pain': 4,
        'Lethargic': 5,
        'Unresponsive': 6
    },
    'Goal Richmond-RAS Scale': {
        ' 0  Alert and calm': 0,
        '-1 Awakens to voice (eye opening/contact) > 10 sec': 1,
        '-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec': 2,
        '-3 Moderate sedation, movement or eye opening; No eye contact': 3,
        '-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation': 4,
        '-5 Unarousable, no response to voice or physical stimulation': 5
    },
    'Richmond-RAS Scale': {
        '+4 Combative, violent, danger to staff': 0,
        '+3 Pulls or removes tube(s) or catheter(s); aggressive': 1,
        '+2 Frequent nonpurposeful movement, fights ventilator': 2,
        '+1 Anxious, apprehensive, but not aggressive': 3,
        ' 0  Alert and calm': 4,
        '-1 Awakens to voice (eye opening/contact) > 10 sec': 5,
        '-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec': 6,
        '-3 Moderate sedation, movement or eye opening; No eye contact': 7,
        '-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation': 8,
        '-5 Unarousable, no response to voice or physical stimulation': 9
    },
    'GCS - Motor Response': {
        'Obeys Commands': 0,
        'Localizes Pain': 1,
        'Flex-withdraws': 2,
        'Abnormal Flexion': 3,
        'Abnormal extension': 4,
        'No response': 5
    },
    'GCS - Verbal Response': {
        'Oriented': 0,
        'Confused': 1,
        'Inappropriate Words': 2,
        'Incomprehensible sounds': 3,
        'No Response': 4,
        'No Response-ETT': 5
    },
    'GCS - Eye Opening': {
        'Spontaneously': 0,
        'To Speech': 1,
        'To Pain': 2
    }
}


for label, mapping in category_maps.items():
    mask = filtered_chartevents_df['label'] == label
    filtered_chartevents_df.loc[mask, 'valuenum'] = filtered_chartevents_df.loc[mask, 'value'].map(mapping)
    
vital_signs = [
    'Heart Rate',
    'Non Invasive Blood Pressure systolic',
    'Non Invasive Blood Pressure diastolic',
    'Arterial Blood Pressure systolic',
    'Arterial Blood Pressure diastolic',
    'Central Venous Pressure',
]


vital_signs += [
    'Respiratory Rate',
    'Peak Insp. Pressure',
    'Minute Volume',
    'Apnea Interval',
    'PEEP set',
    'Tidal Volume (observed)',
    'Mean Airway Pressure',
    'Inspired O2 Fraction',
    'O2 Flow',
]

vital_signs += [
    'GCS - Eye Opening',
    'GCS - Verbal Response',
    'GCS - Motor Response',
    'Richmond-RAS Scale',
    'Goal Richmond-RAS Scale',
    'Level of Consciousness',
]


vital_signs += [
    'Temperature Fahrenheit'
]


vital_signs += [
    'O2 saturation pulseoxymetry',
    'Inspired O2 Fraction',
    'O2 Flow'
]

filtered_chartevents_vital = filtered_chartevents_df[filtered_chartevents_df['label'].isin(vital_signs)]
filtered_chartevents_vital = filtered_chartevents_vital.dropna(subset=['valuenum'])

subject_id_indices = [i for i, col in enumerate(df_features.columns) if col == 'subject_id']

if len(subject_id_indices) > 1:
    cols_to_keep = [i for i in range(len(df_features.columns)) if i not in subject_id_indices[1:]]
    df_features = df_features.iloc[:, cols_to_keep]


# load
df_vitalsign = pd.read_csv('data/vitalsign.csv.gz')
df_vitalsign = df_vitalsign[df_vitalsign['subject_id'].isin(df_features['subject_id'].unique())]
df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

new_df_vital = []

for subject_id in tqdm(df_vitalsign['subject_id'].unique()):
    df_vital_subject = df_vitalsign[df_vitalsign['subject_id'] == subject_id]
    df_diag_subject = df_features[df_features['subject_id'] == subject_id]
    
    for _, diag_row in df_diag_subject.iterrows():
        mask = (df_vital_subject['charttime'] >= diag_row['icu_intime']) & \
               (df_vital_subject['charttime'] <= diag_row['icu_outtime'])
        new_df_vital.append(df_vital_subject[mask])

new_df_vital = pd.concat(new_df_vital, ignore_index=True)
new_df_vital = new_df_vital.drop_duplicates()

vital_map = {
    'temperature': {'label': 'Temperature Fahrenheit', 'itemid': 223761, 'valueuom': 'F'},
    'heartrate': {'label': 'Heart Rate', 'itemid': 220045, 'valueuom': 'bpm'},
    'resprate': {'label': 'Respiratory Rate', 'itemid': 220210, 'valueuom': 'insp/min'},
    'o2sat': {'label': 'O2 saturation pulseoxymetry', 'itemid': 220277, 'valueuom': '%'},
    'sbp': {'label': 'Non Invasive Blood Pressure systolic', 'itemid': 220179, 'valueuom': 'mmHg'},
    'dbp': {'label': 'Non Invasive Blood Pressure diastolic', 'itemid': 220180, 'valueuom': 'mmHg'}}

new_rows = []

for _, row in new_df_vital.iterrows():
    for col, config in vital_map.items():
        val = row[col]
        if pd.notna(val):
            new_row = {
                'subject_id': row['subject_id'],
                'hadm_id': None,
                'stay_id': row['stay_id'],
                'caregiver_id': None,
                'charttime': pd.to_datetime(row['charttime']),
                'storetime': pd.to_datetime(row['charttime']),
                'itemid': config['itemid'],
                'value': None,
                'valuenum': float(val),
                'valueuom': config['valueuom'],
                'warning': None,
                'label': config['label']
            }
            new_rows.append(new_row)

new_vitals_df = pd.DataFrame(new_rows)

filtered_chartevents_vital = pd.concat([filtered_chartevents_vital, new_vitals_df], ignore_index=True)

filtered_chartevents_vital = filtered_chartevents_vital[['subject_id','charttime','storetime','valuenum','valueuom','label']]

absolute_threshold = 0.8e6 

cleaned_vitals_list = []

for label in tqdm(filtered_chartevents_vital['label'].unique()):
    df_label = filtered_chartevents_vital[filtered_chartevents_vital['label'] == label]
    df_label_cleaned = df_label[df_label['valuenum'].notna()]

    lower_quantile = df_label_cleaned['valuenum'].quantile(0.001)
    upper_quantile = df_label_cleaned['valuenum'].quantile(0.999)

    df_label_cleaned = df_label_cleaned[
        (df_label_cleaned['valuenum'] >= lower_quantile) &
        (df_label_cleaned['valuenum'] <= upper_quantile) &
        (df_label_cleaned['valuenum'] < absolute_threshold)
    ]

    cleaned_vitals_list.append(
        df_label_cleaned[['subject_id','charttime','storetime','valuenum','valueuom','label']]
    )

filtered_chartevents_vital = pd.concat(cleaned_vitals_list, ignore_index=True)

output_dir_raw = 'preprocessed_vitalsigns/raw_data/'
os.makedirs(output_dir_raw, exist_ok=True)

vital_grouped = filtered_chartevents_vital.groupby('subject_id')

for subject_id, df_vital_subject in tqdm(vital_grouped, total=len(vital_grouped)):
    df_vital_subject.to_csv(f'{output_dir_raw}{subject_id}.csv', index=False)
    
input_dir_raw = 'preprocessed_vitalsigns/raw_data/'
all_labels = set()

csv_files = [f for f in os.listdir(input_dir_raw) if f.endswith('.csv')]

for csv_file in tqdm(csv_files, total=len(csv_files)):
    df = pd.read_csv(os.path.join(input_dir_raw, csv_file), usecols=['label'])
    all_labels.update(df['label'].dropna().unique())

unique_vitalsigns = np.array(sorted(all_labels))

unique_vitalsigns = ['Apnea Interval', 'Arterial Blood Pressure diastolic',
       'Arterial Blood Pressure systolic', 'Central Venous Pressure',
       'GCS - Eye Opening', 'GCS - Motor Response',
       'GCS - Verbal Response', 'Goal Richmond-RAS Scale', 'Heart Rate',
       'Inspired O2 Fraction', 'Level of Consciousness',
       'Mean Airway Pressure', 'Minute Volume',
       'Non Invasive Blood Pressure diastolic',
       'Non Invasive Blood Pressure systolic', 'O2 Flow',
       'O2 saturation pulseoxymetry', 'PEEP set', 'Peak Insp. Pressure',
       'Respiratory Rate', 'Richmond-RAS Scale', 'Temperature Fahrenheit',
       'Tidal Volume (observed)']

input_dir_raw = 'preprocessed_vitalsigns/raw_data/'
output_dir_stats = 'preprocessed_vitalsigns/stats_data2/'
os.makedirs(output_dir_stats, exist_ok=True)
output_dir_stats = 'preprocessed_vitalsigns/stats_data2/'

batch_files = [f for f in os.listdir(output_dir_stats) if f.startswith('batch_') and f.endswith('_vitals.csv')]
batch_files = natsorted(batch_files)  

batch_size = 5000

if batch_files:
    last_file = int(batch_files[-1].split('_')[-2])
    file_count = last_file + 1
    start = file_count * batch_size
else:
    file_count = 0
    start = 0
    

buffer = []

raw_keys = ['min','max','first','last',
            'hours_after_min','hours_after_max',
            'hours_after_first','hours_after_last']

diff_keys = raw_keys.copy()

for index, row in tqdm(df_features.iloc[start:, :40].iterrows(),
                       total=len(df_features) - start,
                       initial=start,
                      leave=False):

    subject_id = row['subject_id']
    context_start = pd.to_datetime(row['context_start'])
    context_end = pd.to_datetime(row['context_end'])
    context_length_hours = (context_end - context_start).total_seconds() / 3600

    raw_data_path = f'{input_dir_raw}{subject_id}.csv'

    row_features = {
        'subject_id': subject_id,
        'context_start': context_start,
        'context_end': context_end
    }

    if os.path.exists(raw_data_path):

        df_vitals_subject = pd.read_csv(raw_data_path)
        df_vitals_subject['charttime'] = pd.to_datetime(df_vitals_subject['charttime'], errors='coerce', format="mixed",)
        df_vitals_subject['storetime'] = pd.to_datetime(df_vitals_subject['storetime'], errors='coerce', format="mixed",)

        filtered_vitals = df_vitals_subject[
            (df_vitals_subject['charttime'] >= context_start) &
            (df_vitals_subject['charttime'] <= context_end) &
            (df_vitals_subject['storetime'] >= context_start) &
            (df_vitals_subject['storetime'] <= context_end)
        ]

        for vitalsign in unique_vitalsigns:

            fs = filtered_vitals[filtered_vitals['label'] == vitalsign].sort_values('storetime')
            vs_clean = vitalsign.lower().replace('-', '_').replace(' ', '_') \
                                   .replace('___', '_').replace('__', '_')

            if not fs.empty:

                values = fs['valuenum'].dropna().values
                times_in_hours = (fs['storetime'] - context_start).dt.total_seconds() / 3600

                stats_raw = compute_stats(values, times_in_hours, context_length_hours, include_hours_before_end=True)

                diffs = compute_diffs(values)
                diffs_times = times_in_hours[1:] if len(times_in_hours) > 1 else None
                stats_diff = compute_stats(diffs, diffs_times, context_length_hours, include_hours_before_end=True)

                for k, v in stats_raw.items():
                    row_features[f'vitalsigns_raw_{vs_clean}_{k}'] = v
                for k, v in stats_diff.items():
                    row_features[f'vitalsigns_diffs_{vs_clean}_{k}'] = v

            else:
                for k in raw_keys:
                    row_features[f'vitalsigns_raw_{vs_clean}_{k}'] = np.nan
                for k in diff_keys:
                    row_features[f'vitalsigns_diffs_{vs_clean}_{k}'] = np.nan

    else:
        for vitalsign in unique_vitalsigns:
            vs_clean = vitalsign.lower().replace('-', '_').replace(' ', '_') \
                                   .replace('___', '_').replace('__', '_')
            
            for k in raw_keys:
                row_features[f'vitalsigns_raw_{vs_clean}_{k}'] = np.nan
            for k in diff_keys:
                row_features[f'vitalsigns_diffs_{vs_clean}_{k}'] = np.nan

    buffer.append(row_features)

    if len(buffer) >= batch_size:
        pd.DataFrame(buffer).to_csv(f'{output_dir_stats}batch_{file_count}_vitals.csv', index=False)
        buffer = []
        file_count += 1


if buffer:
    pd.DataFrame(buffer).to_csv(f'{output_dir_stats}batch_{file_count}_vitals.csv', index=False)
    
output_dir_stats = 'preprocessed_vitalsigns/stats_data2/'

batch_files = [f for f in os.listdir(output_dir_stats) if f.startswith('batch_') and f.endswith('_vitals.csv')]
batch_files = natsorted(batch_files)  

all_batches = []

for file in tqdm(batch_files, desc="Loading batch files"):
    batch_df = pd.read_csv(os.path.join(output_dir_stats, file))
    all_batches.append(batch_df)

combined_stats_df = pd.concat(all_batches, ignore_index=True)

df_features = pd.concat(
    [df_features.reset_index(drop=True), combined_stats_df.reset_index(drop=True)], 
    axis=1
)

df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

df_features = df_features.rename(columns={
    'age': 'demographics_age',
    'gender': 'demographic_gender'
})

race_dummies = pd.get_dummies(df_features['race_mapped'], prefix='demographics_race')
race_dummies.columns = [
    'demographics_race_WHITE',
    'demographics_race_BLACK',
    'demographics_race_HISPANIC',
    'demographics_race_ASIAN',
    'demographics_race_OTHER'
]

df_features = df_features.drop(columns=['race_mapped'])
df_features = pd.concat([df_features, race_dummies], axis=1)

race_dummies = None
df_features = df_features.loc[:, ~df_features.columns.duplicated()]


# load
omr = pd.read_csv('data/omr.csv.gz')
omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')
omr['chartdate'] = pd.to_datetime(omr['chartdate'])
omr = omr[omr['result_name'].isin(['BMI (kg/m2)', 'Height (Inches)', 'Weight (Lbs)'])]

omr = omr[omr['subject_id'].isin(df_features['subject_id'].unique())]
omr.dropna(subset=['result_value'], inplace=True)

conditions = [
    (omr['result_name'] == 'BMI (kg/m2)') & (omr['result_value'] > 100),
    (omr['result_name'] == 'Weight (Lbs)') & (omr['result_value'] > 880),   
    (omr['result_name'] == 'Weight (Lbs)') & (omr['result_value'] < 44),     
    (omr['result_name'] == 'Height (Inches)') & (omr['result_value'] > 157), 
    (omr['result_name'] == 'Height (Inches)') & (omr['result_value'] < 24)   
]

for condition in conditions:
    omr.loc[condition, 'result_value'] = np.nan

omr.dropna(subset=['result_value'], inplace=True)


out_bmi = []
out_weight = []
out_height = []

for _, row in tqdm(df_features.iterrows()):
    patient = row['subject_id']
    intime = row['ecg_time']
    
    df_patient = omr[omr['subject_id'] == patient]
    df_within = df_patient[
        (df_patient['chartdate'] >= (intime - pd.Timedelta(days=180))) &
        (df_patient['chartdate'] <= (intime + pd.Timedelta(days=180)))
    ]
    
    if df_within.empty:
        out_bmi.append(np.nan)
        out_weight.append(np.nan)
        out_height.append(np.nan)
    else:
        
        bmi_rows = df_within[df_within['result_name'] == 'BMI (kg/m2)']
        out_bmi.append(
            bmi_rows.iloc[(bmi_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            if not bmi_rows.empty else np.nan
        )
        
        weight_rows = df_within[df_within['result_name'] == 'Weight (Lbs)']
        out_weight.append(
            weight_rows.iloc[(weight_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            if not weight_rows.empty else np.nan
        )
        
        height_rows = df_within[df_within['result_name'] == 'Height (Inches)']
        out_height.append(
            height_rows.iloc[(height_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
            if not height_rows.empty else np.nan
        )

df_features['biometrics_bmi'] = out_bmi
df_features['biometrics_weight'] = out_weight
df_features['biometrics_height'] = out_height

df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

df_features.drop(columns='row_index', inplace=True, errors='ignore')
df_features.drop(columns='ed_diag_ed', inplace=True, errors='ignore')
df_features.drop(columns='ed_diag_hosp', inplace=True, errors='ignore')
df_features.drop(columns='hosp_diag_hosp', inplace=True, errors='ignore')
df_features.drop(columns='all_diag_hosp', inplace=True, errors='ignore')
df_features.drop(columns='all_diag_all', inplace=True, errors='ignore')

df_features = df_features.rename(columns={'Unnamed: 0': 'data'})
df_features = df_features.rename(columns={'Diagnoses_labels': 'diagnoses_labels'})
df_features.drop(columns='race', inplace=True, errors='ignore')
df_features.drop(columns='fold', inplace=True, errors='ignore')
df_features.drop(columns='ecg_taken_in_ed', inplace=True, errors='ignore')
df_features.drop(columns='ecg_taken_in_hosp', inplace=True, errors='ignore')
df_features.drop(columns='ecg_taken_in_ed_or_hosp', inplace=True, errors='ignore')
df_features = df_features.rename(columns={'demographic_gender': 'demographics_gender'})
df_features.drop(columns='anchor_age', inplace=True, errors='ignore')
df_features.drop(columns='anchor_year', inplace=True, errors='ignore')

lbl_diags = np.load('lbl_diags.npy')

chunks = []
chunk_size = 10000

for i in tqdm(range(0, len(df_features), chunk_size)):
    chunk = df_features['diagnoses_labels'].iloc[i:i+chunk_size].apply(pd.Series)
    chunk.columns = [f'diagnoses_{label}' for label in lbl_diags]
    chunks.append(chunk)

diagnoses_df = pd.concat(chunks, ignore_index=True)

df_features.drop(columns='diagnoses_labels', inplace=True)
df_features = pd.concat([df_features, diagnoses_df], axis=1)
diagnoses_df = None
df_features.to_pickle('df_features2.pkl')
df_features = pd.read_pickle('df_features2.pkl')


df_features['mortality_hours'] = (df_features['dod'] - df_features['context_end']).dt.total_seconds() / 3600

df_features['deterioration_mortality_icustay'] = (
    pd.notnull(df_features['dod']) &
    (df_features['dod'] >= df_features['icu_intime']) &
    (df_features['dod'] <= df_features['icu_outtime'])
).astype(int)

df_features['deterioration_mortality_stay'] = (
    pd.notnull(df_features['dod']) &
    (df_features['dod'] >= df_features['icu_intime']) &
    (df_features['dod'] <= df_features['hosp_dischtime'])
).astype(int)

mortality_time_windows = {
    'deterioration_mortality_1D': 24,
    'deterioration_mortality_2D': 48,
    'deterioration_mortality_7D': 168,
    'deterioration_mortality_28D': 672,
    'deterioration_mortality_90D': 2160,
    'deterioration_mortality_180D': 4320,
    'deterioration_mortality_365D': 8760
}

for col, time_window in mortality_time_windows.items():
    df_features[col] = ((df_features['mortality_hours'] <= time_window) & 
                        pd.notnull(df_features['mortality_hours'])).astype(float)
    
df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')
df_features['delta_icu'] = df_features['icu_outtime'] - df_features['context_end']
df_features['delta_hosp'] = df_features['hosp_dischtime'] - df_features['context_end']

for days in tqdm([1, 2, 7]):
    df_features[f'deterioration_discharge_icu_{days}day'] = (
        df_features['delta_icu'] <= pd.Timedelta(days=days)
    ).astype('float')

    df_features[f'deterioration_discharge_hosp_{days}day'] = (
        df_features['delta_hosp'] <= pd.Timedelta(days=days)
    ).astype('float')
    
df_features.drop(columns=['delta_icu', 'delta_hosp'], inplace=True)
df_features['ecg_hours_since_ecg'] = (df_features['context_end'] - df_features['ecg_time']).dt.total_seconds() / 3600


# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)
# load
inputevents = pd.read_csv('data/inputevents.csv.gz', compression='gzip', low_memory=False)


inputevents = inputevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')
top_500_labels = inputevents['label'].value_counts().head(500).index.tolist()
filtered_inputevents = inputevents[inputevents['label'].isin(top_500_labels)]
filtered_inputevents = filtered_inputevents[filtered_inputevents['subject_id'].isin(df_features['subject_id'].unique())]
inputevents = None
d_items = None

crystalloids = [
    'NaCl 0.9%', 'Dextrose 5%', 'Free Water', 'LR', 'D5NS', 'D5 1/2NS', 'D5LR', 'D5 1/4NS',
    'NaCl 0.45%', 'Sterile Water', 'Dextrose 10%', 'Dextrose 20%', 'Dextrose 30%',
    'Dextrose 40%', 'Dextrose 50%', 'NaCl 3% (Hypertonic Saline)', 'NaCl 23.4%'
]

electrolytes = [
    'Potassium Chloride', 'KCL (Bolus)', 'KCl (CRRT)', 'K Phos', 'Na Phos', 'Calcium Gluconate',
    'Calcium Gluconate (CRRT)', 'Calcium Gluconate (Bolus)', 'Calcium Chloride', 
    'Magnesium Sulfate', 'Magnesium Sulfate (Bolus)', 'Magnesium Sulfate (OB-GYN)',
    'Sodium Bicarbonate 8.4%', 'Sodium Bicarbonate 8.4% (Amp)', 'Hydrochloric Acid - HCL'
]

antibiotics = [
    'Cefepime', 'Vancomycin', 'Ceftriaxone', 'Levofloxacin', 'Azithromycin', 'Metronidazole',
    'Bactrim (SMX/TMP)', 'Cefazolin', 'Ciprofloxacin', 'Meropenem', 'Piperacillin/Tazobactam (Zosyn)',
    'Piperacillin', 'Omeprazole (Prilosec)', 'Tobramycin', 'Doxycycline', 'Linezolid',
    'Daptomycin', 'Ceftazidime', 'Ampicillin/Sulbactam (Unasyn)', 'Ampicillin', 'Acyclovir',
    'Clindamycin', 'Aztreonam', 'Colistin', 'Amikacin', 'Imipenem/Cilastatin', 'Ceftaroline',
    'Rifampin', 'Erythromycin', 'Gentamicin', 'Nafcillin', 'Tamiflu', 'Penicillin G potassium',
    'Keflex', 'Quinine', 'Isoniazid', 'Ethambutol', 'Pyrazinamide'
]


vasopressors = [
    'Epinephrine', 'Epinephrine.', 'Norepinephrine', 'Vasopressin',
    'Dobutamine', 'Dopamine', 'Phenylephrine', 'Phenylephrine (50/250)', 'Phenylephrine (200/250)',
    'Isuprel', 'Angiotensin II (Giapreza)'
]

inotropes = [
    'Epinephrine', 'Epinephrine.', 'Dobutamine', 'Dopamine',
    'Isuprel', 'Milrinone'
]

antiarrhythmics = [
    'Amiodarone', 'Amiodarone 600/500', 'Amiodarone 450/250', 'Amiodarone 150/100',
    'Esmolol', 'Lidocaine', 'Procainamide', 'Verapamil', 'Diltiazem', 'Adenosine'
]

anticoagulants_antiplatelets = [
    'Heparin Sodium', 'Heparin Sodium (Prophylaxis)', 'Heparin Sodium (Impella)', 
    'Heparin Sodium (CRRT-Prefilter)', 'Enoxaparin (Lovenox)', 'Bivalirudin (Angiomax)', 
    'Eptifibatide (Integrilin)', 'Coumadin (Warfarin)', 'Argatroban', 'Fondaparinux',
    'Tirofiban (Aggrastat)', 'Abciximab (Reopro)', 'Lepirudin', 'ACD-A Citrate (1000ml)', 
    'ACD-A Citrate (500ml)', 'Citrate', 'Protamine sulfate'
]

sedatives = [
    'Propofol', 'Midazolam (Versed)', 'Lorazepam (Ativan)', 'Diazepam (Valium)', 
    'Dexmedetomidine (Precedex)', 'Ketamine', 'Pentobarbital'
]

analgesics = [
    'Fentanyl', 'Fentanyl (Concentrate)', 'Morphine Sulfate', 'Hydromorphone (Dilaudid)',
    'Meperidine (Demerol)', 'Acetaminophen-IV', 'Methadone Hydrochloride', 
    'Ketorolac (Toradol)', 'Naloxone (Narcan)'
]

neuromuscular_blockers = [
    'Vecuronium', 'Rocuronium', 'Cisatracurium', 'Neostigmine (Prostigmin)'
]

gi_protection = [
    'Ranitidine (Prophylaxis)', 'Pantoprazole (Protonix)', 'Famotidine (Pepcid)',
    'Lansoprazole (Prevacid)', 'Omeprazole (Prilosec)', 'Carafate (Sucralfate)', 
    'Esomeprazole (Nexium)', 'Ranitidine'
]

blood_products_transfusions = [
    'Packed Red Blood Cells (pRBCs)', 'Platelets', 'Fresh Frozen Plasma (FFP)',
    'Cryoprecipitate', 'Whole Blood', 'Albumin 5%', 'Albumin 25%', 'IVIG (Intravenous Immunoglobulin)',
    'Factor VIII', 'Factor IX', 'Prothrombin Complex Concentrate (PCC)', 'Recombinant Factor VIIa',
    'Fibrinogen Concentrate', 'Thrombin', 'Tranexamic Acid (TXA)', 'Erythropoietin (EPO)',
    'Iron Sucrose (Venofer)', 'Iron Dextran', 'Iron Gluconate', 'Ferumoxytol'
]

parenteral_nutrition = [
    'TPN w/ Lipids', 'TPN without Lipids', 'Peripheral Parenteral Nutrition', 
    'Dextrose PN', 'Amino Acids', 'Lipids 20%', 'Lipids 10%', 'Lipids (additive)'
]

groups = {
    'crystalloids': crystalloids,
    'electrolytes': electrolytes,
    'antibiotics': antibiotics,
    'vasopressors': vasopressors,
    'inotropes': inotropes,
    'antiarrhythmics': antiarrhythmics,
    'anticoagulants_antiplatelets': anticoagulants_antiplatelets,
    'sedatives': sedatives,
    'analgesics': analgesics,
    'neuromuscular_blockers': neuromuscular_blockers,
    'gi_protection': gi_protection,
    'blood_products_transfusions': blood_products_transfusions,
    'parenteral_nutrition': parenteral_nutrition
}

filtered_inputevents['endtime'] = pd.to_datetime(filtered_inputevents['endtime'])
filtered_inputevents['starttime'] = pd.to_datetime(filtered_inputevents['starttime'])

df_features['context_end'] = pd.to_datetime(df_features['context_end'])

all_group_labels = [drug for drugs in groups.values() for drug in drugs]
filtered_inputevents = filtered_inputevents[filtered_inputevents['label'].isin(all_group_labels)].copy()

df_features['row_id'] = df_features.index
df_features['context_end_plus_24h'] = df_features['context_end'] + timedelta(hours=24)

for group_name in groups:
    df_features[f'deterioration_medications_{group_name}'] = 0

for group_name, drug_list in tqdm(groups.items()):
    group_events = filtered_inputevents[filtered_inputevents['label'].isin(drug_list)].copy()

    merged = df_features[['row_id', 'subject_id', 'context_end', 'context_end_plus_24h']].merge(
        group_events[['subject_id', 'starttime', 'endtime']],
        on='subject_id',
        how='inner'
    )

    mask = (
        (merged['starttime'] < merged['context_end_plus_24h']) &
        (merged['endtime'] > merged['context_end'])
    )

    matched_row_ids = merged.loc[mask, 'row_id'].unique()
    df_features.loc[df_features['row_id'].isin(matched_row_ids), f'deterioration_medications_{group_name}'] = 1

df_features.drop(columns=['row_id', 'context_end_plus_24h'], inplace=True)

df_features.to_pickle('df_features2.pkl')


df_features = pd.read_pickle('df_features2.pkl')

d_small = df_features[['subject_id', 'context_end']].copy()

severe_hypoxemia = []

vitals_cache = {}

for row in tqdm(d_small.itertuples(index=False), total=len(d_small)):
    subject_id = row.subject_id
    context_end = pd.to_datetime(row.context_end)

    if subject_id not in vitals_cache:
        file_path = f'preprocessed_vitalsigns/raw_data/{subject_id}.csv'
        if os.path.exists(file_path):
            try:
             
                df_vital = pd.read_csv(
                    file_path,
                    usecols=['label', 'charttime', 'storetime', 'valuenum']
                )
  
                df_vital = df_vital[df_vital['label'] == 'O2 saturation pulseoxymetry'].copy()

                df_vital['charttime'] = pd.to_datetime(df_vital['charttime'], errors='coerce', format="mixed")
                df_vital['storetime'] = pd.to_datetime(df_vital['storetime'], errors='coerce', format="mixed")

                vitals_cache[subject_id] = df_vital

            except Exception as e:
                print(f"Error loading file for subject {subject_id}: {e}")
                vitals_cache[subject_id] = None
        else:
            vitals_cache[subject_id] = None

    df_vital = vitals_cache[subject_id]

    if df_vital is None or df_vital.empty:
        severe_hypoxemia.append(0)
        continue

    mask = (
        ((df_vital['charttime'] >= context_end) & (df_vital['charttime'] <= context_end + timedelta(hours=24))) |
        ((df_vital['storetime'] >= context_end) & (df_vital['storetime'] <= context_end + timedelta(hours=24)))
    )
    df_window = df_vital[mask]

    if not df_window.empty and (df_window['valuenum'] <= 85).any():
        severe_hypoxemia.append(1)
    else:
        severe_hypoxemia.append(0)
        
df_features['deterioration_severe_hypoxemia'] = severe_hypoxemia
df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')


# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)
d_items_subset = d_items[['itemid', 'label']]

# load
procedures = pd.read_csv('data/procedureevents.csv.gz')
procedures = procedures.merge(d_items_subset, on='itemid', how='left')

mechanical_ventilation_invasive_procedures = ["Invasive Ventilation","Intubation"]
mechanical_ventilation_noninvasive_procedures = ["Non-invasive Ventilation"]

procedures = procedures[procedures['label'].isin(mechanical_ventilation_invasive_procedures+mechanical_ventilation_noninvasive_procedures)]

procedures['starttime'] = pd.to_datetime(procedures['starttime'])
procedures['endtime'] = pd.to_datetime(procedures['endtime'])

mechanical_ventilation_inv = []
mechanical_ventilation_noninv = []

d_small = df_features[['subject_id', 'context_end', 'context_start']].copy()

for row in tqdm(d_small.itertuples(index=False), total=len(d_small)):
    subject_id = row.subject_id
    c_start = row.context_start
    c_end = row.context_end

    proc_patient = procedures[procedures['subject_id'] == subject_id]

    if proc_patient.empty:
        mechanical_ventilation_inv.append(0)
        mechanical_ventilation_noninv.append(0)
        continue

    overlap_mask = (
        (proc_patient['starttime'] <= c_end) &
        (proc_patient['endtime'] >= c_start)
    )
    
    mv_inv = proc_patient[overlap_mask & 
                          proc_patient['label'].isin(mechanical_ventilation_invasive_procedures)]
    mechanical_ventilation_inv.append(int(not mv_inv.empty))

    mv_noninv = proc_patient[overlap_mask &
                             proc_patient['label'].isin(mechanical_ventilation_noninvasive_procedures)]
    mechanical_ventilation_noninv.append(int(not mv_noninv.empty))
    
df_features['mechanical_ventilation_invasive'] = mechanical_ventilation_inv
df_features['mechanical_ventilation_noninvasive'] = mechanical_ventilation_noninv
df_features.to_pickle('df_features2.pkl')



df_features = pd.read_pickle('df_features2.pkl')


# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)
    
filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]

# load
procedures = pd.read_csv('data/procedureevents.csv.gz')
procedures = procedures.merge(d_items_subset, on='itemid', how='left')

ecmo_procedures = ["Sheath (Venous)"]
mechanical_ventilation_invasive_procedures = ["Invasive Ventilation", "Intubation"]
mechanical_ventilation_noninvasive_procedures = ["Non-invasive Ventilation"]

procedures = procedures[procedures['label'].isin(ecmo_procedures+mechanical_ventilation_invasive_procedures+mechanical_ventilation_noninvasive_procedures)]

ecmo_chartevents = ["Circuit Configuration (ECMO)",
                    "Speed (ECMO)",
                    "Flow (ECMO)",
                    "Sweep (ECMO)",
                    "Flow Alarm (Lo) (ECMO)",
                    "Flow Alarm (Hi) (ECMO)",
                    "FiO2 (ECMO)",
                    "Suction events (ECMO)",
                    "Cannula sites visually inspected (ECMO)",
                    "Oxygenator visible (ECMO)",
                    "Pump plugged into RED outlet (ECMO)",
                    "Circuit inspected for clot (ECMO)",
                    "P1 - P2 (ECMO)",
                    "P1 (ECMO)",
                    "P2 (ECMO)",
                    "Emergency Equipment at bedside (ECMO)",
                    "Flow Sensor repositioned (ECMO)",
                    "Oxygenator/ECMO",
                    "ECMO"]


filtered_chartevents_df = filtered_chartevents_df[filtered_chartevents_df['label'].isin(ecmo_chartevents)]
procedures['starttime'] = pd.to_datetime(procedures['starttime'])
procedures['endtime'] = pd.to_datetime(procedures['endtime'])
filtered_chartevents_df['charttime'] = pd.to_datetime(filtered_chartevents_df['charttime'])
filtered_chartevents_df['storetime'] = pd.to_datetime(filtered_chartevents_df['storetime'])

ecmo = []
mechanical_ventilation_inv = []
mechanical_ventilation_noninv = []

df_features['context_end_plus_24h'] = df_features['context_end'] + timedelta(hours=24)

d_small = df_features[['subject_id', 'context_end', 'context_end_plus_24h']].copy()

for row in tqdm(d_small.itertuples(index=False), total=len(d_small)):
    subject_id = row.subject_id
    context_end = row.context_end
    context_end_plus_24h = row.context_end_plus_24h

    proc_patient = procedures[procedures['subject_id'] == subject_id]
    chart_patient = filtered_chartevents_df[filtered_chartevents_df['subject_id'] == subject_id]

    ecmo_proc = proc_patient[proc_patient['label'].isin(ecmo_procedures)]
    ecmo_chart = chart_patient[chart_patient['label'].isin(ecmo_chartevents)]

    ecmo_found = (
        ((ecmo_proc['starttime'] < context_end_plus_24h) & (ecmo_proc['endtime'] > context_end)).any() |
        ((ecmo_chart['charttime'] >= context_end) & (ecmo_chart['charttime'] <= context_end_plus_24h) &
         (ecmo_chart['storetime'] >= context_end) & (ecmo_chart['storetime'] <= context_end_plus_24h)).any()
    )
    ecmo.append(1 if ecmo_found else 0)

    mv_proc_inv = proc_patient[proc_patient['label'].isin(mechanical_ventilation_invasive_procedures)]

    mv_inv_found = (
        ((mv_proc_inv['starttime'] < context_end_plus_24h) & 
         (mv_proc_inv['endtime'] > context_end)).any()
    )
    mechanical_ventilation_inv.append(1 if mv_inv_found else 0)

    mv_proc_noninv = proc_patient[proc_patient['label'].isin(mechanical_ventilation_noninvasive_procedures)]

    mv_noninv_found = (
        ((mv_proc_noninv['starttime'] < context_end_plus_24h) & 
         (mv_proc_noninv['endtime'] > context_end)).any()
    )
    
    mechanical_ventilation_noninv.append(1 if mv_noninv_found else 0)
    
df_features['deterioration_ecmo'] = ecmo
df_features['deterioration_mechanical_ventilation_invasive'] = mechanical_ventilation_inv
df_features['deterioration_mechanical_ventilation_noninvasive'] = mechanical_ventilation_noninv
df_features.to_pickle('df_features2.pkl')


df_features = pd.read_pickle('df_features2.pkl')
# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)
d_items_subset = d_items[['itemid', 'label']]
# load
procedures = pd.read_csv('data/procedureevents.csv.gz')

procedures = procedures.merge(d_items_subset, on='itemid', how='left')
cardiac_arrests = procedures[procedures['itemid']==225466]
cardiac_arrests['starttime'] = pd.to_datetime(cardiac_arrests['starttime'])
cardiac_arrests['endtime'] = pd.to_datetime(cardiac_arrests['endtime'])

cardiacs = []

d_small = df_features[['subject_id', 'context_end', 'context_end_plus_24h']].copy()

for row in tqdm(d_small.itertuples(index=False), total=len(d_small)):
    subject_id = row.subject_id
    context_end = row.context_end
    context_end_plus_24h = row.context_end_plus_24h

    proc_patient = cardiac_arrests[cardiac_arrests['subject_id'] == subject_id]

    if proc_patient.empty:
        cardiacs.append(0)
        continue

    arrest_found = (
        (proc_patient['starttime'] < context_end_plus_24h) &
        (proc_patient['endtime'] > context_end)
    ).any()

    cardiacs.append(1 if arrest_found else 0)

df_features['deterioration_cardiac_arrest'] = cardiacs

df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

output_dir_raw = 'preprocessed_labvalues/raw_data/'
arterial_o2_dfs = []

for filename in tqdm(os.listdir(output_dir_raw)):
    if filename.endswith('.csv'):
        subject_id = filename.replace('.csv', '') 
        file_path = os.path.join(output_dir_raw, filename)
        df = pd.read_csv(file_path)
        filtered = df[df['label'] == 'Arterial O2 pressure']
        if not filtered.empty:
            filtered['subject_id'] = int(subject_id)  
            arterial_o2_dfs.append(filtered)

df_arterial_o2 = pd.concat(arterial_o2_dfs, ignore_index=True)

df_arterial_o2 = df_arterial_o2[df_arterial_o2['valuenum'] >= 0].copy()

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]

fi02_events = filtered_chartevents_df[filtered_chartevents_df['itemid']==223835]
fi02_events = fi02_events[fi02_events['valuenum'] <= 100].copy()
fi02_events = fi02_events[~((fi02_events['valuenum'] > 0) & (fi02_events['valuenum'] < 1))].copy()
fi02_events['valuenum'] = fi02_events['valuenum'] / 100.0

# load
procedures = pd.read_csv('data/procedureevents.csv.gz')
procedures = procedures.merge(d_items_subset, on='itemid', how='left')

mechanical_ventilation_procedures = ["Invasive Ventilation",
                                     "Non-invasive Ventilation",
                                     "Intubation"]

procedures_mechanical = procedures[procedures['label'].isin(mechanical_ventilation_procedures)]

procedures_mechanical['starttime'] = pd.to_datetime(procedures_mechanical['starttime'])
procedures_mechanical['endtime'] = pd.to_datetime(procedures_mechanical['endtime'])

mechanical_ventilation_chartevents = ["Ventilator Type",
                                      "Ventilator Mode",
                                      "Ventilator Mode (Hamilton)",
                                      "PEEP set",
                                      "Total PEEP Level",
                                      "Ventilator Tank #1",
                                      "Ventilator Tank #2",
                                      "Respiratory Rate (Set)",
                                      "Respiratory Rate (Total)"]

chartevents_mechanical = filtered_chartevents_df[filtered_chartevents_df['label'].isin(mechanical_ventilation_chartevents)]

chartevents_mechanical['charttime'] = pd.to_datetime(chartevents_mechanical['charttime'])
chartevents_mechanical['storetime'] = pd.to_datetime(chartevents_mechanical['storetime'])

df_arterial_o2['charttime'] = pd.to_datetime(df_arterial_o2['charttime'])
fi02_events['charttime'] = pd.to_datetime(fi02_events['charttime'])

df_arterial_o2 = df_arterial_o2.drop(columns=['label', 'storetime'], errors='ignore')

fi02_events = fi02_events.drop(columns=[
    'hadm_id', 'stay_id', 'caregiver_id', 'storetime',
    'itemid', 'valueuom', 'warning', 'label'
], errors='ignore')

procedures_mechanical = procedures_mechanical[['subject_id','starttime','endtime']]
chartevents_mechanical = chartevents_mechanical[['subject_id','charttime','storetime']]

empty_pao2_df = pd.DataFrame(columns=['subject_id', 'charttime', 'valuenum'])
empty_fio2_df = pd.DataFrame(columns=['subject_id', 'charttime', 'valuenum'])
empty_proc_df = pd.DataFrame(columns=['subject_id', 'starttime', 'endtime'])
empty_chart_df = pd.DataFrame(columns=['subject_id', 'charttime', 'storetime'])

def resample_hourly(df, time_col='charttime', val_col='valuenum'):
    df = df.sort_values(time_col)
    df['hour_bin'] = pd.cut(df[time_col], bins=hourly_bins, right=False)
    last_vals = df.groupby('hour_bin', observed=True)[val_col].min()
    last_vals = last_vals.reindex(pd.IntervalIndex.from_breaks(hourly_bins, closed='left'))
    return last_vals.ffill()

output_dir = 'sofa/respiratory/'
os.makedirs(output_dir, exist_ok=True)

batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

batch_size = 20000

if batch_files:
    last_file = int(batch_files[-1].split('_')[-2])
    file_count = last_file + 1
    start = file_count * batch_size
else:
    file_count = 0
    start = 0

arterial_o2_dict = dict(tuple(df_arterial_o2.groupby('subject_id')))
fio2_dict = dict(tuple(fi02_events.groupby('subject_id')))
proc_mech_dict = dict(tuple(procedures_mechanical.groupby('subject_id')))
chart_mech_dict = dict(tuple(chartevents_mechanical.groupby('subject_id')))

buffer = []

for row in tqdm(df_features.iloc[start:,:40][['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    pao2_values = arterial_o2_dict.get(subject_id, empty_pao2_df)
    fio2_values = fio2_dict.get(subject_id, empty_fio2_df)
    proc_mech = proc_mech_dict.get(subject_id, empty_proc_df)
    chart_mech = chart_mech_dict.get(subject_id, empty_chart_df)

    pao2_values = pao2_values[(pao2_values['charttime'] > context_end) & (pao2_values['charttime'] < context_end_24)]
    fio2_values = fio2_values[(fio2_values['charttime'] > context_end) & (fio2_values['charttime'] < context_end_24)]
    
    on_ventilation = False
    proc_mech_window = proc_mech[
        ((proc_mech['starttime'] >= context_end) & (proc_mech['starttime'] < context_end_24)) |
        ((proc_mech['endtime'] >= context_end) & (proc_mech['endtime'] < context_end_24))
    ]
    if not proc_mech_window.empty:
        on_ventilation = True

    if not on_ventilation:
        chart_mech_window = chart_mech[
            ((chart_mech['charttime'] >= context_end) & (chart_mech['charttime'] < context_end_24)) |
            ((chart_mech['storetime'] >= context_end) & (chart_mech['storetime'] < context_end_24))
        ]
        if not chart_mech_window.empty:
            on_ventilation = True

    if not pao2_values.empty and not fio2_values.empty:
        hourly_bins = pd.date_range(start=context_end, end=context_end_24, freq='H')

        pao2_hourly = resample_hourly(pao2_values)
        fio2_hourly = resample_hourly(fio2_values)

        ratios = pao2_hourly / fio2_hourly
        worst_ratio = ratios.min(skipna=True)

        if pd.notna(worst_ratio):
            if worst_ratio < 100 and on_ventilation:
                score = 1
            elif worst_ratio < 200 and on_ventilation:
                score = 1
            elif worst_ratio < 300:
                score = 1
            else:
                score = 0
        else:
            score = -999

        buffer.append(score)
    else:
        buffer.append(-999)

    if len(buffer) >= batch_size:
        batch_df = pd.DataFrame({'score': buffer})
        batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)
        buffer = []
        file_count += 1

if buffer:
    batch_df = pd.DataFrame({'score': buffer})
    batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)
    
batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

all_batches = []

for file in tqdm(batch_files, desc="Loading batch files"):
    batch_df = pd.read_csv(os.path.join(output_dir, file))
    all_batches.append(batch_df)

combined_stats_df = pd.concat(all_batches, ignore_index=True)

combined_stats_df.columns = ['deterioration_sofa_respiratory']
df_features['deterioration_sofa_respiratory'] = combined_stats_df['deterioration_sofa_respiratory'].values
df_features.to_pickle('df_features2.pkl')


df_features = pd.read_pickle('df_features2.pkl')
# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]
filtered_chartevents_df['charttime'] = pd.to_datetime(filtered_chartevents_df['charttime'])
gcs_chartevents = filtered_chartevents_df[filtered_chartevents_df['label'].isin(['GCS - Eye Opening', 
                                                                                'GCS - Verbal Response',
                                                                                'GCS - Motor Response'])]
gcs_chartevents = gcs_chartevents[['subject_id','charttime','storetime','valuenum','label']]

gcs_eye = gcs_chartevents[gcs_chartevents['label']=='GCS - Eye Opening']
gcs_verbal = gcs_chartevents[gcs_chartevents['label']=='GCS - Verbal Response']
gcs_motor = gcs_chartevents[gcs_chartevents['label']=='GCS - Motor Response']

empty_gcs = pd.DataFrame(columns=['subject_id',
                                  'charttime',
                                  'storetime',
                                  'valuenum',
                                 'label'])

eye_dict = dict(tuple(gcs_eye.groupby('subject_id')))
verbal_dict = dict(tuple(gcs_verbal.groupby('subject_id')))
motor_dict = dict(tuple(gcs_motor.groupby('subject_id')))

output_dir = 'sofa/nervous/'
os.makedirs(output_dir, exist_ok=True)

batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

batch_size = 20000

if batch_files:
    last_file = int(batch_files[-1].split('_')[-2])
    file_count = last_file + 1
    start = file_count * batch_size
else:
    file_count = 0
    start = 0
    
def resample_hourly(df):
    df = df.sort_values('charttime')
    df['hour_bin'] = pd.cut(df['charttime'], bins=hourly_bins, right=False)
    last_vals = df.groupby('hour_bin', observed=True)['valuenum'].min()
    last_vals = last_vals.reindex(pd.IntervalIndex.from_breaks(hourly_bins, closed='left'))
    return last_vals.ffill()

deterioration_sofa_nervous = []

for row in tqdm(df_features.iloc[start:, :40][['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    eye_values = eye_dict.get(subject_id, empty_gcs)
    verbal_values = verbal_dict.get(subject_id, empty_gcs)
    motor_values = motor_dict.get(subject_id, empty_gcs)

    eye_values = eye_values[(eye_values['charttime'] > context_end) & (eye_values['charttime'] < context_end_24)]
    verbal_values = verbal_values[(verbal_values['charttime'] > context_end) & (verbal_values['charttime'] < context_end_24)]
    motor_values = motor_values[(motor_values['charttime'] > context_end) & (motor_values['charttime'] < context_end_24)]

    missing_count = sum([eye_values.empty, verbal_values.empty, motor_values.empty])
    if missing_count >= 1:
        deterioration_sofa_nervous.append(-999)
    else:
        hourly_bins = pd.date_range(start=context_end, end=context_end_24, freq='H')
        eye_hourly = resample_hourly(eye_values)
        verbal_hourly = resample_hourly(verbal_values)
        motor_hourly = resample_hourly(motor_values)

        gcs_total = eye_hourly + verbal_hourly + motor_hourly

        valid_counts = (~eye_hourly.isna()).astype(int) + \
                       (~verbal_hourly.isna()).astype(int) + \
                       (~motor_hourly.isna()).astype(int)

        gcs_total[valid_counts < 3] = np.nan

        if gcs_total.isna().all():
            deterioration_sofa_nervous.append(-999)
        else:
            min_idx = gcs_total.idxmin()
            min_score = gcs_total.min()

            if (
                not pd.isna(eye_hourly[min_idx]) and
                not pd.isna(verbal_hourly[min_idx]) and
                not pd.isna(motor_hourly[min_idx])
            ):
                deterioration_sofa_nervous.append(1 if min_score <= 12 else 0)
            else:
                deterioration_sofa_nervous.append(-999)

    if len(deterioration_sofa_nervous) == batch_size:
        batch_df = pd.DataFrame({'score': deterioration_sofa_nervous})
        batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)
        deterioration_sofa_nervous = []
        file_count += 1


if deterioration_sofa_nervous:
    batch_df = pd.DataFrame({'score': deterioration_sofa_nervous})
    batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)

output_dir = 'sofa/nervous/'

batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

all_batches = []

for file in tqdm(batch_files, desc="Loading batch files"):
    batch_df = pd.read_csv(os.path.join(output_dir, file))
    all_batches.append(batch_df)

combined_stats_df = pd.concat(all_batches, ignore_index=True)

combined_stats_df.columns = ['deterioration_sofa_nervous']

df_features['deterioration_sofa_nervous'] = combined_stats_df['deterioration_sofa_nervous'].values

df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
 
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')

    filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]

maps = ['Arterial Blood Pressure mean',
        'Non Invasive Blood Pressure mean']

chart_maps = filtered_chartevents_df[filtered_chartevents_df['label'].isin(maps)]

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)
# load
inputevents = pd.read_csv('data/inputevents.csv.gz', compression='gzip', low_memory=False)

inputevents = inputevents.merge(d_items[['itemid', 'label']], on='itemid', how='left')
cardiovascular = ['Dopamine','Dobutamine','Epinephrine','Norepinephrine']
cardio_events = inputevents[inputevents['label'].isin(cardiovascular)]

inputevents = None
filtered_chartevents_df = None

empty_cardio = pd.DataFrame(columns=['subject_id',
                                  'starttime',
                                  'endtime',
                                  'valuenum',
                                 'label'])

cardio_events = cardio_events[['subject_id','starttime','endtime','label']]

cardio_events['starttime'] = pd.to_datetime(cardio_events['starttime'])
cardio_events['endtime'] = pd.to_datetime(cardio_events['endtime'])

cardio_dict = dict(tuple(cardio_events.groupby('subject_id')))

output_dir = 'sofa/cardiovascular/'
os.makedirs(output_dir, exist_ok=True)

batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

batch_size = 20000

if batch_files:
    last_file = int(batch_files[-1].split('_')[-2])
    file_count = last_file + 1
    start = file_count * batch_size
else:
    file_count = 0
    start = 0
    
deterioration_sofa_cardio = []

for row in tqdm(df_features.iloc[start:,:40][['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    cardio_values = cardio_dict.get(subject_id, empty_cardio)
    
    overlapping = cardio_values[
                    (cardio_values['starttime'] < context_end_24) &
                    (cardio_values['endtime'] > context_end)
                ]

    deterioration_sofa_cardio.append(1 if not overlapping.empty else 0)
            
    if len(deterioration_sofa_cardio) >= batch_size:
        batch_df = pd.DataFrame({'score': deterioration_sofa_cardio})
        batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)
        deterioration_sofa_cardio = []
        file_count += 1

if deterioration_sofa_cardio:
    batch_df = pd.DataFrame({'score': deterioration_sofa_cardio})
    batch_df.to_csv(f'{output_dir}batch_{file_count}_stats.csv', index=False)
    
output_dir = 'sofa/cardiovascular/'

batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_stats.csv')]
batch_files = natsorted(batch_files)

all_batches = []

for file in tqdm(batch_files, desc="Loading batch files"):
    batch_df = pd.read_csv(os.path.join(output_dir, file))
    all_batches.append(batch_df)

combined_stats_df = pd.concat(all_batches, ignore_index=True)

combined_stats_df.columns = ['deterioration_sofa_cardiovascular']
df_features['deterioration_sofa_cardiovascular'] = combined_stats_df['deterioration_sofa_cardiovascular'].values
df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    
    filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')
    filtered_chunk = filtered_chunk[filtered_chunk['label']=='Total Bilirubin']
        
    if not filtered_chunk.empty:
        filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]

bilirubin_events = filtered_chartevents_df[filtered_chartevents_df['label']=='Total Bilirubin']
bilirubin_events = bilirubin_events[bilirubin_events['valuenum']<0.8e6]
bilirubin_events = bilirubin_events[['subject_id','charttime','storetime','itemid','valuenum','valueuom','label']]
bilirubin_events = bilirubin_events[bilirubin_events['valueuom']=='mg/dL']
filtered_chartevents_df = None

empty_liver = pd.DataFrame(columns=['subject_id',
                                  'charttime',
                                  'storetime',
                                  'valuenum',
                                 'label'])

bilirubin_events['charttime'] = pd.to_datetime(bilirubin_events['charttime'])
liver_dict = dict(tuple(bilirubin_events.groupby('subject_id')))

deterioration_sofa_liver = []

for row in tqdm(df_features[['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    liver_values = liver_dict.get(subject_id, empty_liver)
    
    overlapping = liver_values[
                    (liver_values['charttime'] < context_end_24) &
                    (liver_values['charttime'] > context_end)
                ]
    
    if not overlapping.empty and (overlapping['valuenum'] >= 2).any():
        deterioration_sofa_liver.append(1)
    else:
        deterioration_sofa_liver.append(0)
        

df_features['deterioration_sofa_liver'] = deterioration_sofa_liver
df_features.to_pickle('df_features2.pkl')

df_features = pd.read_pickle('df_features2.pkl')

# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    if not filtered_chunk.empty:
        filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')
        filtered_chunk = filtered_chunk[filtered_chunk['label']=='Platelet Count']
        filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]

platelet_events = filtered_chartevents_df[filtered_chartevents_df['label']=='Platelet Count']

platelet_events = platelet_events[platelet_events['valuenum']<0.8e6]
platelet_events = platelet_events[['subject_id','charttime','storetime','itemid','valuenum','valueuom','label']]
platelet_events = platelet_events[platelet_events['valueuom']=='K/uL']
filtered_chartevents_df = None
empty_coagulation = pd.DataFrame(columns=['subject_id',
                                  'charttime',
                                  'storetime',
                                  'valuenum',
                                 'label'])

platelet_events['charttime'] = pd.to_datetime(platelet_events['charttime'])
coagulation_dict = dict(tuple(platelet_events.groupby('subject_id')))

deterioration_sofa_coagulation = []

for row in tqdm(df_features[['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    platelet_values = coagulation_dict.get(subject_id, empty_coagulation)
    
    overlapping = platelet_values[
                    (platelet_values['charttime'] < context_end_24) &
                    (platelet_values['charttime'] > context_end)
                ]
    
    if not overlapping.empty and (overlapping['valuenum'] < 100).any():
        deterioration_sofa_coagulation.append(1)
    else:
        deterioration_sofa_coagulation.append(0)
df_features['deterioration_sofa_coagulation'] = deterioration_sofa_coagulation
df_features.to_pickle('df_features2.pkl')

# load
df_features = pd.read_pickle('df_features2.pkl')
# load
d_items = pd.read_csv('data/d_items.csv.gz', compression='gzip', low_memory=False)

chunksize = 1000000
filename = 'data/chartevents.csv.gz'

relevant_subject_ids = set(df_features['subject_id'].unique())
relevant_stay_ids = set(df_features['icu_stay_id'].unique())

filtered_chunks = []

d_items_subset = d_items[['itemid', 'label']]

chartevents_iter = pd.read_csv(
    filename,
    compression='gzip',
    low_memory=True,
    chunksize=chunksize
)

for chunk in tqdm(chartevents_iter):
    filtered_chunk = chunk[
        (chunk['subject_id'].isin(relevant_subject_ids)) &
        (chunk['stay_id'].isin(relevant_stay_ids))
    ]
    
    filtered_chunk = filtered_chunk.merge(d_items_subset, on='itemid', how='left')
    filtered_chunk = filtered_chunk[filtered_chunk['label']=='Creatinine (serum)']
        
    if not filtered_chunk.empty:
        filtered_chunks.append(filtered_chunk)

filtered_chartevents_df = pd.concat(filtered_chunks, ignore_index=True)
filtered_chartevents_df = filtered_chartevents_df[~(filtered_chartevents_df['label'] == 'Safety Measures')]
creatinine_events = filtered_chartevents_df[filtered_chartevents_df['label']=='Creatinine (serum)']
creatinine_events = creatinine_events[creatinine_events['valuenum']<0.8e6]

creatinine_events = creatinine_events[['subject_id','charttime','storetime','itemid','valuenum','valueuom','label']]
creatinine_events = creatinine_events[creatinine_events['valueuom']=='mg/dL']

# load
outputevents = pd.read_csv('data/outputevents.csv.gz', compression='gzip', low_memory=False)
outputevents = outputevents.merge(d_items_subset, on='itemid', how='left')
urine_labels = ['Foley','Void','Condom Cath','OR Urine','PACU Urine']
urine_events = outputevents[outputevents['label'].isin(urine_labels)]
filtered_chartevents_df = None
urine_events = urine_events[['subject_id','charttime','value','label']]
empty_kidneys = pd.DataFrame(columns=['subject_id',
                                  'charttime',
                                  'storetime',
                                  'valuenum',
                                 'label'])
creatinine_events['charttime'] = pd.to_datetime(creatinine_events['charttime'])
kidneys_dict = dict(tuple(creatinine_events.groupby('subject_id')))
empty_urine = pd.DataFrame(columns=['subject_id',
                                  'charttime',
                                  'value',
                                  'label'])

urine_events['charttime'] = pd.to_datetime(urine_events['charttime'])
urine_events = urine_events[urine_events['subject_id'].isin(df_features['subject_id'].unique())]
urine_dict = dict(tuple(urine_events.groupby('subject_id')))
deterioration_sofa_kidneys = []

for row in tqdm(df_features[['subject_id', 'context_start', 'context_end']].itertuples(index=False)):

    subject_id = row.subject_id
    context_end = row.context_end
    context_end_24 = context_end + pd.Timedelta(hours=24)

    creatinine_values = kidneys_dict.get(subject_id, empty_kidneys)
    urine_events = urine_dict.get(subject_id, empty_urine)
    
    overlapping = creatinine_values[
                    (creatinine_values['charttime'] < context_end_24) &
                    (creatinine_values['charttime'] > context_end)
                ]
    
    overlapping_urine = urine_events[
                    (urine_events['charttime'] < context_end_24) &
                    (urine_events['charttime'] > context_end)
                ]
    
    total_urine_ml = overlapping_urine['value'].sum()
    
    if (not overlapping.empty and (overlapping['valuenum'] >= 2.0).any()) or (total_urine_ml < 500):
        deterioration_sofa_kidneys.append(1)
    else:
        deterioration_sofa_kidneys.append(0)

df_features['deterioration_sofa_kidneys'] = deterioration_sofa_kidneys
df_features.to_pickle('df_features2.pkl')


df_features = pd.read_pickle('df_features2.pkl')

cols = [c for c in df_features.columns if c.startswith("deterio")]

df_features[cols] = df_features[cols].replace(-999, np.nan)

df_features = df_features.drop(
    columns=df_features.filter(regex=r'^diagnoses').columns
)

df_features = df_features.drop(
    columns=df_features.filter(regex=r'deterioration_discharge').columns
)

df_rest = df_features[~df_features['strat_fold'].isin([18, 19])].copy()
df_val  = df_features[df_features['strat_fold'] == 18].copy()
df_test = df_features[df_features['strat_fold'] == 19].copy()

det_cols = [c for c in df_features.columns if c.startswith('deterioration')]
det_cols = [c for c in det_cols if 'deterioration_discharge' not in c]
MIN_POS = 3

def check_feasible(df, cols, min_pos):
    return [c for c in cols if df[c].sum() < min_pos]

bad_val  = check_feasible(df_val,  det_cols, MIN_POS)
bad_test = check_feasible(df_test, det_cols, MIN_POS)

if bad_val or bad_test:
    raise ValueError(
        f"Infeasible deterioration constraints.\n"
        f"Val bad cols: {len(bad_val)}\n"
        f"Test bad cols: {len(bad_test)}"
    )
    
def constrained_sample_one_per_icu(df, cols, min_pos):
    selected_idx = []
    pos_counter = Counter()

    groups = df.groupby(
        ['subject_id', 'icu_stay_id', 'icu_intime'],
        sort=False
    )

    for _, group in tqdm(groups, desc="Sampling ICU stays"):
        best_idx = None
        best_gain = -1

        for idx, row in group.iterrows():
            gain = 0
            for c in cols:
                if row[c] == 1 and pos_counter[c] < min_pos:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        selected_idx.append(best_idx)

        # update counters
        for c in cols:
            if df.loc[best_idx, c] == 1:
                pos_counter[c] += 1

    return df.loc[selected_idx].copy()

df_val  = constrained_sample_one_per_icu(df_val,  det_cols, MIN_POS)
df_test = constrained_sample_one_per_icu(df_test, det_cols, MIN_POS)

def verify_constraints(df, cols, min_pos):
    return [c for c in cols if df[c].sum() < min_pos]

bad_val_after  = verify_constraints(df_val,  det_cols, MIN_POS)
bad_test_after = verify_constraints(df_test, det_cols, MIN_POS)

if bad_val_after or bad_test_after:
    raise RuntimeError(
        f"Constraint violation after sampling.\n"
        f"Val bad cols: {bad_val_after}\n"
        f"Test bad cols: {bad_test_after}"
    )
    

val_icu_ids  = set(df_val['icu_stay_id'].unique())
test_icu_ids = set(df_test['icu_stay_id'].unique())

overlap = val_icu_ids & test_icu_ids

df_features = pd.concat(
    [df_rest, df_val, df_test],
    axis=0
).reset_index(drop=True)

all_labels = [i for i in df_features.columns if 'deterioration_' in i]

services_hadm = set(df_services['hadm_id'].dropna().unique())

features_hadm = set(df_features['hosp_hadm_id'].dropna().unique())
features_icu  = set(df_features['icu_stay_id'].dropna().unique())

overlap_hadm = services_hadm & features_hadm
overlap_icu  = services_hadm & features_icu

# load
df_services = pd.read_csv('data/services.csv.gz')
df_services['transfertime'] = pd.to_datetime(df_services['transfertime'])

unique_surgs = [i for i in df_services['curr_service'].unique() if 'SURG' in i]
df_services = df_services[df_services['curr_service'].isin(unique_surgs)]

df_features['context_end'] = pd.to_datetime(df_features['context_end'])

for surg in unique_surgs:
    df_features[f'surgery_{surg}_24'] = 0
    df_features[f'surgery_{surg}_stay'] = 0

for idx, row in tqdm(df_features.iloc[:,:40].iterrows(), total=len(df_features)):
    subject_id = row['subject_id']
    hosp_hadm_id = row['hosp_hadm_id']
    context_end = row['context_end']

    df_patient = df_services[df_services['subject_id'] == subject_id]

    for surg in unique_surgs:
        df_surg = df_patient[df_patient['curr_service'] == surg]

        if ((df_surg['hadm_id'] == hosp_hadm_id).any()):
            df_features.at[idx, f'surgery_{surg}_stay'] = 1

        if ((df_surg['transfertime'] >= context_end - pd.Timedelta(hours=24)) &
            (df_surg['transfertime'] <= context_end)).any():
            df_features.at[idx, f'surgery_{surg}_24'] = 1
            
            
demographics_columns = [i for i in df_features.columns if 'demographics_' in i]
biometrics_columns = [i for i in df_features.columns if 'biometrics_' in i]
vitalparameters_columns = [i for i in df_features.columns if 'vitalsigns_' in i]
labvalues_columns = [i for i in df_features.columns if 'labvalues_' in i]
surgery_columns = [i for i in df_features.columns if 'surgery_' in i]
ecg_columns = [i for i in df_features.columns if 'ecg_hours_since_ecg' in i]
mechanical_columns = [i for i in df_features.columns if i.startswith("mechanical_")]

all_features = demographics_columns + biometrics_columns + vitalparameters_columns + labvalues_columns + surgery_columns + ecg_columns + mechanical_columns

df_features[all_features] = df_features[all_features].astype(float)
df_features[all_labels] = df_features[all_labels].astype(float)

df_features[all_features] = df_features[all_features].mask(df_features[all_features] > 3000, np.nan)

cols_to_drop = df_features[all_features].columns[df_features[all_features].isna().all()]
df_features = df_features.drop(columns=cols_to_drop)

demographics_columns = [i for i in df_features.columns if 'demographics_' in i]
biometrics_columns = [i for i in df_features.columns if 'biometrics_' in i]
vitalparameters_columns = [i for i in df_features.columns if 'vitalsigns_' in i]
labvalues_columns = [i for i in df_features.columns if 'labvalues_' in i]
surgery_columns = [i for i in df_features.columns if 'surgery_' in i]
ecg_columns = [i for i in df_features.columns if 'ecg_hours_since_ecg' in i]
mechanical_columns = [i for i in df_features.columns if i.startswith("mechanical_")]

all_features = demographics_columns + biometrics_columns + vitalparameters_columns + labvalues_columns + surgery_columns + ecg_columns + mechanical_columns

df_rest = None
df_subset = None
df_test = None

np.save('data/memmap/lbl_itos.npy', np.array(all_labels))

df_features.to_pickle('data/memmap/mds_icu2.pkl')

df_features = pd.read_pickle('data/memmap/mds_icu2.pkl')
df_ecg = df_features[df_features['ecg_hours_since_ecg']==0]
col_dets = [i for i in df_ecg.columns if 'deterioration' in i]

df_features = df_ecg

demographics_cols = [i for i in df_features.columns if 'demographics_' in i]
biometrics_cols = [i for i in df_features.columns if 'biometrics_' in i]
vitalparameters_cols = [i for i in df_features.columns if 'vitalsigns_' in i]
labvalues_cols = [i for i in df_features.columns if 'labvalues_' in i]
surgery_cols = [i for i in df_features.columns if 'surgery_' in i]
ecg_cols = [i for i in df_features.columns if 'ecg_hours_since_ecg' in i]
mechanical_cols = [i for i in df_features.columns if i.startswith("mechanical_")]

all_features = (
    demographics_cols + biometrics_cols + vitalparameters_cols +
    labvalues_cols + surgery_cols + ecg_cols + mechanical_cols
)

mask_dict = {f"{col}_mask": df_features[col].isna().astype(int) for col in all_features}
mask_df = pd.DataFrame(mask_dict, index=df_features.index)


train_mask = df_features['strat_fold'].between(0, 17)
for col in all_features:
    median_val = df_features.loc[train_mask, col].median()
    df_features[col] = df_features[col].fillna(median_val)

df_features = pd.concat([df_features, mask_df], axis=1)

mask_df = None
mask_dict = None

allfeatures_mask = [f"{col}_mask" for col in all_features]
total_features = all_features + allfeatures_mask

df_features["features"] = df_features[total_features].to_numpy().tolist()

df_features.drop(columns=total_features, inplace=True)

deterioration_labels = [i for i in df_features.columns if 'deterioration_' in i]
lbl_itos = deterioration_labels

df_features["label"] = df_features[lbl_itos].to_numpy().tolist()

df_features.drop(columns=lbl_itos, inplace=True)

df_features.to_pickle('data/memmap/mds_icu_preprocessed2_ecg.pkl')
np.save('data/memmap/lbl_itos.npy', lbl_itos)
