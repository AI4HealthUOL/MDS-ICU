__all__ = ['get_available_channels', 'channel_stoi_default', 'channel_stoi_canonical', 'resample_data', 'get_filename_out', 'reformat_data_ptb',
           'prepare_data_ptb', 'mat_to_np', 'reformat_data_icbeb', 'prepare_data_icbeb', 'prepare_data_ptb_xl', 'prepare_data_sph',
           'map_and_filter_labels', 'thew_to_np', 'reformat_data_thew', 'prepare_data_thew', 'prepare_data_cinc',
           'prepare_data_chapman', 'prepare_data_ribeiro_test', 'prepare_data_ludb', 'prepare_data_ribeiro_full',
           'prepare_mitdb','prepare_afdb','prepare_ltafdb','prepare_vfdb','prepare_cpsc21','prepare_icentia','get_data_annot_stats',
           'prepare_mimicecg','prepare_ptbv2','prepare_heedb']

import traceback
import wfdb

import scipy.io

#icentia
import os
import shutil
import zipfile

import numpy as np
import pandas as pd

import resampy
from tqdm.auto import tqdm
from pathlib import Path

#thew
from ishneholterlib import Holter
from stratify import stratify,stratify_batched

#ribeiro
import h5py
import datetime

#from clinical_ts.misc_utils import *
from clinical_ts.data.time_series_dataset_utils import *

from sklearn.model_selection import StratifiedKFold

from concurrent.futures import ProcessPoolExecutor, as_completed

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}
channel_stoi_canonical = {"i": 0, "ii": 1, "iii":2, "avr":3, "avl":4, "avf":5, "v1":6, "v2":7, "v3":8, "v4":9, "v5":10, "v6":11, "vx":12, "vy":13, "vz":14}

def get_stratified_kfolds(labels,n_splits,random_state):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    return skf.split(np.zeros(len(labels)),labels)

def get_available_channels(channel_labels, channel_stoi):
    if(channel_stoi is None):
        return range(len(channel_labels))
    else:
        return sorted([channel_stoi[c] for c in channel_labels if c in channel_stoi.keys()])

def fix_nans_and_clip(signal,clip_amp=3):
    for i in range(signal.shape[1]):
        tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
        signal[:,i]= np.clip(tmp,a_max=clip_amp, a_min=-clip_amp) if clip_amp>0 else tmp
    return signal

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                #if(skimage_transform):
                #    data[:,channel_stoi[cl]]=transform.resize(sigbufs[:,i],(timesteps_new,),order=interpolation_order).astype(np.float32)
                #else:
                #    data[:,channel_stoi[cl]]=zoom(sigbufs[:,i],timesteps_new/len(sigbufs),order=interpolation_order).astype(np.float32)
                data[:,channel_stoi[cl]] = resampy.resample(sigbufs[:,i], fs, target_fs).astype(np.float32)
    else:
        #if(skimage_transform):
        #    data=transform.resize(sigbufs,(timesteps_new,channels),order=interpolation_order).astype(np.float32)
        #else:
        #    data=zoom(sigbufs,(timesteps_new/len(sigbufs),1),order=interpolation_order).astype(np.float32)
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data


def get_filename_out(filename_in, target_folder=None, suffix=""):
    if target_folder is None:
        #absolute path here
        filename_out = filename_in.parent/(filename_in.stem+suffix+".npy")
        filename_out_relative = filename_out
    else:
        if("train" in filename_in.parts):
            target_folder_train = target_folder/"train"
            # relative path here
            filename_out = target_folder_train/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)

            target_folder_train.mkdir(parents=True, exist_ok=True)
        elif("eval" in filename_in.parts or "dev_test" in filename_in.parts or "valid" in filename_in.parts or "valtest" in filename_in.parts):
            target_folder_valid = target_folder/"valid"
            filename_out = target_folder_valid/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder_valid.mkdir(parents=True, exist_ok=True)
        else:
            filename_out = target_folder/(filename_in.stem+suffix+".npy")
            filename_out_relative = filename_out.relative_to(target_folder)
            target_folder.mkdir(parents=True, exist_ok=True)
    return filename_out, filename_out_relative


def _age_to_categorical(age):
    if(np.isnan(age)):
        label_age = -1
    elif(age<30):
        label_age = 0
    elif(age<40):
        label_age = 1
    elif(age<50):
        label_age = 2
    elif(age<60):
        label_age = 3
    elif(age<70):
        label_age = 4
    elif(age<80):
        label_age = 5
    else:
        label_age = 6
    return label_age

def _sex_to_categorical(sex):
    sex_mapping = {"n/a":-1, "male":0, "female":1, "":-1}
    return sex_mapping[sex]

def reformat_data_ptb(datafiles, target_fs=200, channels=12, channel_stoi=channel_stoi_canonical, lbl_itos=None, target_folder=None):
    rows = []
    for filename_in in tqdm(datafiles):
        filename_out, filename_out_relative = get_filename_out(filename_in,target_folder)

        sigbufs, header = wfdb.rdsamp(str(filename_in.parent/filename_in.stem))
        data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs, channels=channels)
        np.save(filename_out,data.astype(np.float32))

        metadata=[(":".join(h.split(":")[1:])).strip() for h in header['comments']]
        metadata_descr = [(h.split(":")[0]).strip() for h in header['comments']]
        #for i,(m1,m2) in enumerate(zip(metadata_descr,metadata)):
        #    print(i,m1,m2)
        #input()
        ecg = filename_in.stem
        patient = str(filename_in.parent)[-3:]
        channels_available = get_available_channels(header['sig_name'],channel_stoi)
        try:
            age=int(metadata[0])
        except ValueError:
            age=np.nan
        label_age = _age_to_categorical(age)

        diagnosis1 = metadata[4]
        diagnosis2 = [l.strip() for l in  metadata[7].split(",") if l.strip() not in ["no","unknown","n/a"]]
        diagnosis2 = [l.replace("Typ IIa","Type IIa").replace("Hypertension","hypertension").replace(" (intraventricular gradient 100-160mmHg)","").replace("Fibrillation","fibrillation").replace("Recurrent ventricular tachycardias","Recurrent ventricular tachycardia").replace("Thyriodectomy","Thyroidectomy").replace("Dilated Cardiomyopathy secondary to coronary artery disease. Mitral regurgitation (grade 2)","Dilated Cardiomyopathy") for l in diagnosis2]#clean up
        #remove type from Hyperlipoproteinemia Type xx TODO
        diagnosis2 = [l.replace("Hyperlipoproteinemia Type IIa","Hyperlipoproteinemia").replace("Hyperlipoproteinemia Type IV","Hyperlipoproteinemia").replace("Hyperlipoproteinemia Type IIb","Hyperlipoproteinemia") for l in diagnosis2]#clean up

        diagnosis12 = [diagnosis1]+diagnosis2

        smoker = metadata[8]
        smoker_mapping = {"no":0, "yes":1, "n/a":-1, "unknown":-1}
        label_smoker = smoker_mapping[smoker]

        sex = metadata[1]
        label_sex = _sex_to_categorical(sex)

        row={"data":filename_out_relative,"dataset":"PTB","channels_available":channels_available,"patient":"PTB_"+patient, "ecg": "PTB_"+ecg, "age": age, "label_age":label_age,"sex": sex, "label_sex": label_sex, "ecg_date":metadata[2], "diagnosis":diagnosis1, "infarction_localization":metadata[5], "diagnosis2": diagnosis2, "diagnosis12": diagnosis12,"smoker": smoker, "label_smoker":label_smoker, "infarction_date":metadata[10], "catheterization_date":metadata[14]}
        #add hemodynamics
        for i in range(14,36):
            row[metadata_descr[i]]=metadata[i]
        #clean up localization
        if(row["diagnosis"]!="Myocardial infarction"):
            row["infarction_localization"]=""
        if row["infarction_localization"] == "infero-latera":
            row["infarction_localization"] = "infero-lateral"
        if row["infarction_localization"] == "infero-posterior-lateral":
            row["infarction_localization"] = "infero-postero-lateral"
        if row["infarction_localization"] == "infero-poster-lateral":
            row["infarction_localization"] = "infero-postero-lateral"
        if row["infarction_localization"] == "no":
            row["infarction_localization"] = ""
        rows.append(row)
    df_all = pd.DataFrame(rows)
    _ptb_cleanup_hemodynamics(df_all)
    #fill label column
    if(lbl_itos is not None):
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        df_all["label"] = df_all.diagnosis.apply(lambda x: (lbl_stoi[x] if x in lbl_stoi.keys() else len(lbl_itos)))#map everything that cannot be found in the dictionary to last entry
    else:
        lbl_itos = np.unique(df_all["diagnosis"])
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        df_all["label"] = df_all.diagnosis.apply(lambda x: lbl_stoi[x])
    #convert datatypes (be careful the standard to_datetime does not work correctly here)
    #df_all.ecg_date=pd.to_datetime(df_all.ecg_date,errors='coerce')
    #df_all.infarction_date=pd.to_datetime(df_all.infarction_date,errors='coerce')
    #df_all.catheterization_date=pd.to_datetime(df_all.catheterization_date,errors='coerce')

    #df_all=df_all.set_index("ecg")
    return df_all, lbl_itos


def _ptb_cleanup_hemodynamics(df_ptb):
    def _extract_numeric(x,mode=0):
        if x=="n/a":
            return np.nan
        else:
            if mode==0:
                return float(x.split(" ")[0])
            elif mode==1:
                return float(x.split(" ")[0].split("/")[0])
            else:
                return float(x.split(" ")[0].split("/")[1])

    def _reformat_col(df, col_in, col_out1, col_out2=None):
        df[col_out1]= df[col_in].apply(lambda x: _extract_numeric(x,0 if col_out2 is None else 1))
        if col_out2 is not None:
            df[col_out2]= df[col_in].apply(lambda x: _extract_numeric(x,2))
        df.drop(col_in,axis=1,inplace=True)

    _reformat_col(df_ptb,"Aorta (at rest) (syst/diast)","aorta_at_rest_syst","aorta_at_rest_diast")
    _reformat_col(df_ptb,"Aorta (at rest) mean","aorta_at_rest_mean")

    df_ptb["Cardiac index (at rest)"]=df_ptb["Cardiac index (at rest)"].apply(lambda x: x.replace(",","."))
    _reformat_col(df_ptb,"Cardiac index (at rest)","cardiac_index_at_rest")
    df_ptb["Cardiac index (load)"]=df_ptb["Cardiac index (load)"].apply(lambda x: x.replace(",","."))
    _reformat_col(df_ptb,"Cardiac index (load)","cardiac_index_load")
    df_ptb["Cardiac output (at rest)"]=df_ptb["Cardiac output (at rest)"].apply(lambda x: x.replace(",","."))
    _reformat_col(df_ptb,"Cardiac output (at rest)","cardiac_output_at_rest")
    df_ptb["Cardiac output (load)"]=df_ptb["Cardiac output (load)"].apply(lambda x: x.replace(",","."))
    _reformat_col(df_ptb,"Cardiac output (load)","cardiac_output_load")

    df_ptb.drop('Chest X-ray',axis=1,inplace=True)
    df_ptb.drop('Left coronary artery stenoses (RCX)',axis=1,inplace=True)
    df_ptb.drop('Left coronary artery stenoses (RIVA)',axis=1,inplace=True)
    df_ptb.drop('Right coronary artery stenoses (RCA)',axis=1,inplace=True)
    df_ptb.drop('Ventriculography',axis=1,inplace=True)
    df_ptb.drop('Catheterization date',axis=1,inplace=True)

    df_ptb["Peripheral blood Pressure (syst/diast)"]=df_ptb["Peripheral blood Pressure (syst/diast)"].apply(lambda x: x.replace("120 /70","120/70"))
    _reformat_col(df_ptb,"Peripheral blood Pressure (syst/diast)","peripheral_bp_syst","peripheral_bp_diast")
    _reformat_col(df_ptb,"Pulmonary artery pressure (at rest) (mean)","pulmonary_ap_at_rest_mean")
    _reformat_col(df_ptb,"Pulmonary artery pressure (at rest) (syst/diast)","pulmonary_ap_at_rest_syst","pulmonary_ap_at_rest_diast")
    _reformat_col(df_ptb,"Pulmonary artery pressure (laod) (mean)","pulmonary_load_mean")
    _reformat_col(df_ptb,"Pulmonary artery pressure (laod) (syst/diast)","pulmonary_ap_load_syst","pulmonary_ap_load_diast")
    _reformat_col(df_ptb,"Pulmonary capillary wedge pressure (at rest)","pulmonary_cwp_at_rest")
    _reformat_col(df_ptb,"Pulmonary capillary wedge pressure (load)","pulmonary_cwp_load")

    df_ptb["Left ventricular enddiastolic pressure"]=df_ptb["Left ventricular enddiastolic pressure"].apply(lambda x:x[-8:].strip())#only post value
    _reformat_col(df_ptb,"Left ventricular enddiastolic pressure","left_ventricular_ed_pressure")


    _reformat_col(df_ptb,"Stroke volume index (at rest)","stroke_volume_index_at_rest")
    df_ptb["Stroke volume index (load)"]=df_ptb["Stroke volume index (load)"].apply(lambda x: x.replace(",","."))
    _reformat_col(df_ptb,"Stroke volume index (load)","stroke_volume_index_load")

def prepare_data_ptb(data_folder, target_folder=None, channel_stoi=channel_stoi_canonical, channels=12, target_fs=50, strat_folds=10, cv_random_state=42,recreate_data=False,ptb_permanently_remove_diagnosis_na = True, ptb_permanently_remove_multiple_ecgs = False,ptb_select_only_first_mi = True, ptb_remove_unknown_loc_mi = True,lbl_itos_ptb_in = ["Healthy control", "Myocardial infarction"],lbl_itos_ptb_regression = ["pulmonary_ap_at_rest_mean"],lbl_itos_ptb_all = ["Diabetes mellitus"]):


    target_root_ptb = Path(".") if target_folder is None else target_folder

    diagnosis_to_int_w_subdiagnosis = {
    #"Healthy control": 0,  # 80
    "anterior": 1,  # 47
    "antero-septal": 1,  # 77
    "antero-septo-lateral": 1,  # 2
    "antero-lateral": 1,  # 43
    "lateral": 1,  # 3
    "inferior": 2,  # 89
    "infero-posterior": 2,  # 1
    "infero-postero-lateral": 2,  # 16
    "infero-poster-lateral": 2,  # 3
    "infero-latera": 2,  # 3
    "infero-lateral": 2,  # 53
    "posterior": 2,  # 4
    "postero-lateral": 2,  # 5
    #"other": -1  # 123
    "":0
    }
    int_to_str=[""," anterior", "inferior"]

    if(recreate_data):
        #collect files
        datafiles_ptb = list(data_folder.glob('**/*.dat'))

        #reformat data
        df_ptb, lbl_itos_ptb=reformat_data_ptb(datafiles=datafiles_ptb,target_fs=target_fs, channels=channels,lbl_itos=lbl_itos_ptb_in, target_folder=target_folder)


        df_ptb["infarction_localization_mapped"]=df_ptb["infarction_localization"].apply(lambda x: int_to_str[diagnosis_to_int_w_subdiagnosis[x]])
        df_ptb["diagnosis_extended"]=df_ptb.apply(lambda row: row["diagnosis"]+" "+row["infarction_localization_mapped"], axis=1)
        #TODO move all this stuff into reformat_data_ptb
        #set regression label(s)
        df_ptb["label_regression"]=df_ptb.apply(lambda row: row[lbl_itos_ptb_regression[0]] if len(lbl_itos_ptb_regression)==1 else [row[regression_col] for regression_col in lbl_itos_ptb_regression] ,axis=1)
        df_ptb.label_regression.apply(lambda x: np.float32(np.log(x)))
        if(len(lbl_itos_ptb_regression)==1):
            df_ptb.label_regression=df_ptb.label_regression.astype(np.float32)

        #set diagnosis_all multilabels
        def extract_multilabel(x, lbl_itos):
            ret = np.zeros(len(lbl_itos),dtype=np.float32)
            for xi in x:
                if(xi in lbl_itos):
                    ret[lbl_itos.index(xi)]=1.0
            return ret

        df_ptb["label_multi_all"]=df_ptb.diagnosis12.apply(lambda x: extract_multilabel(x, lbl_itos_ptb_all) if len(lbl_itos_ptb_all)>1 else int(lbl_itos_ptb_all[0] in x))

        #permanently discard diagnosis n/a
        if(ptb_permanently_remove_diagnosis_na):
            df_ptb = df_ptb[df_ptb.diagnosis != "n/a"].copy()

        #permanently remove multiple ecgs
        if(ptb_permanently_remove_multiple_ecgs):
            df_ptb=df_ptb.sort_values(by='ecg')
            df_ptb["first_ecg"]=df_ptb.ecg.isin(df_ptb.groupby("patient").first().ecg)
            df_ptb["drop"]=df_ptb.ecg.isin(df_ptb[(df_ptb.diagnosis=="Myocardial infarction") & (df_ptb.first_ecg==False)].ecg)
            df_ptb=df_ptb[df_ptb["drop"]==False].reset_index().drop("index",axis=1)

        #discard other diagnosis classes that were not selected
        df_ptb2 = df_ptb[df_ptb.label <= len(lbl_itos_ptb)-1].copy()

        #select first record for MI patients
        if(ptb_select_only_first_mi):
            df_ptb2=df_ptb2.sort_values(by='ecg')
            df_ptb2["first_ecg"]=df_ptb2.ecg.isin(df_ptb2.groupby("patient").first().ecg)
            df_ptb2["drop"]=df_ptb2.ecg.isin(df_ptb2[(df_ptb2.diagnosis=="Myocardial infarction") & (df_ptb2.first_ecg==False)].ecg)
            df_ptb2=df_ptb2[df_ptb2["drop"]==False].reset_index().drop("index",axis=1)
        #remove MIs with unknown infarction status
        if(ptb_remove_unknown_loc_mi):
            df_ptb2=df_ptb2[~((df_ptb2.diagnosis=="Myocardial infarction") & (df_ptb2.infarction_localization==""))]

        #train test split for label (main diagnosis) based on patients
        label_patients=df_ptb2.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        df_ptb["strat_fold"] = -1
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            #now select from ptb2 (as second MIs etc might have been excluded)
            ecgs_valid = np.array(df_ptb2[df_ptb2.patient.isin(patients_valid)]["ecg"])
            df_ptb.loc[df_ptb.ecg.isin(ecgs_valid),"strat_fold"]=i

        #train test split for regression based on ecgs (but does not work properly anyway)
        df_ptb["strat_fold_regression"]=-1
        df_ptb_regression = df_ptb[df_ptb.label_regression.apply(lambda x: np.isfinite(x))].copy()
        label_patients=df_ptb_regression.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            #now select from ptb_regression
            ecgs_valid = np.array(df_ptb_regression[df_ptb_regression.patient.isin(patients_valid)]["ecg"])
            df_ptb.loc[df_ptb.ecg.isin(ecgs_valid),"strat_fold_regression"]=i

        #train test split for label_multi_all (main diagnosis) based on patients
        label_patients=df_ptb.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        df_ptb["strat_fold_all"] = -1
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            df_ptb.loc[df_ptb.patient.isin(patients_valid),"strat_fold_all"]=i

        #train test split for smoker based on patients
        df_ptb_smoker = df_ptb[df_ptb.label_smoker>=0].copy()
        label_patients=df_ptb_smoker.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        df_ptb["strat_fold_smoker"] = -1
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            #now select from ptb_smoker
            ecgs_valid = np.array(df_ptb_smoker[df_ptb_smoker.patient.isin(patients_valid)]["ecg"])
            df_ptb.loc[df_ptb.ecg.isin(ecgs_valid),"strat_fold_smoker"]=i

        #train test split for age based on patients
        df_ptb_age = df_ptb[df_ptb.label_age>=0].copy()
        label_patients=df_ptb_age.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        df_ptb["strat_fold_age"] = -1
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            #now select from ptb_age
            ecgs_valid = np.array(df_ptb_age[df_ptb_age.patient.isin(patients_valid)]["ecg"])
            df_ptb.loc[df_ptb.ecg.isin(ecgs_valid),"strat_fold_age"]=i

        #train test split for sex based on patients
        df_ptb_sex = df_ptb[df_ptb.label_sex>=0].copy()
        label_patients=df_ptb_sex.groupby("patient")["diagnosis_extended"].first()
        splits_patients = get_stratified_kfolds(np.array(label_patients),n_splits=strat_folds,random_state=cv_random_state)
        df_ptb["strat_fold_sex"] = -1
        for i,split in enumerate(splits_patients):
            patients_valid = np.array(label_patients.index)[split[1]]
            #now select from ptb_sex
            ecgs_valid = np.array(df_ptb_sex[df_ptb_sex.patient.isin(patients_valid)]["ecg"])
            df_ptb.loc[df_ptb.ecg.isin(ecgs_valid),"strat_fold_sex"]=i

        #DEBUG
        #importing old splits for debugging
        indexlst = np.load("./old/index_lst.npy")
        evallist = np.load("./old/evallist.npy")
        #trainlist = np.load("./old/trainlist.npy")
        df_ptb["strat_fold_old"] = -1
        for i in range(len(evallist)):
            df_ptb.loc[df_ptb.ecg.isin(["PTB_"+x for x in indexlst[evallist[i]][:,1]]),"strat_fold_old"]=i
        #export current splits for debugging
        evallist =[]
        trainlist =[]
        for i in range(strat_folds):
            valid_ecgs= np.array(df_ptb[(df_ptb.strat_fold>=0)&(df_ptb.strat_fold==i)].ecg.apply(lambda x: x[4:]))
            train_ecgs= np.array(df_ptb[(df_ptb.strat_fold>=0)&(df_ptb.strat_fold!=i)].ecg.apply(lambda x: x[4:]))
            evallist.append([np.where(indexlst==v)[0][0] for v in valid_ecgs])
            trainlist.append([np.where(indexlst==t)[0][0] for t in train_ecgs])
        np.save("./old/evallist_new.npy",evallist)
        np.save("./old/trainlist_new.npy",trainlist)


        #add means and std
        dataset_add_mean_col(df_ptb,data_folder=target_root_ptb)
        dataset_add_std_col(df_ptb,data_folder=target_root_ptb)
        dataset_add_length_col(df_ptb,data_folder=target_root_ptb)
        #dataset_add_median_col(df_ptb,data_folder=target_folder)
        #dataset_add_iqr_col(df_ptb,data_folder=target_folder)

        #compute means and stds
        mean_ptb, std_ptb = dataset_get_stats(df_ptb)

        #save
        save_dataset(df_ptb,lbl_itos_ptb,mean_ptb,std_ptb,target_root_ptb)
    else:
        df_ptb, lbl_itos_ptb, mean_ptb, std_ptb = load_dataset(target_root_ptb,df_mapped=False)
    return df_ptb, lbl_itos_ptb, mean_ptb, std_ptb


def mat_to_np(filename_in, target_fs=100, channels=12, channel_stoi=None, fs=500, target_folder=None):
    channel_labels=["i","ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]

    filename_out, filename_out_relative = get_filename_out(filename_in, target_folder)

    channels_available = get_available_channels(channel_labels,channel_stoi)
    data_raw = scipy.io.loadmat(filename_in)
    sex,age,sigbufs=data_raw['ECG'][0][0]
    sigbufs =np.transpose(sigbufs)

    data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs, channels=channels)
    np.save(filename_out, data.astype(np.float32))
    return sex[0], age[0,0], channels_available, filename_out, filename_out_relative

def reformat_data_icbeb(datafiles, labelfile, target_fs=100, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None):
    #labels
    label_itos=["NORMAL","AF", "I-AVB","LBBB","RBBB","PAC","PVC","STD","STE"]
    labels=pd.read_csv(labelfile)
    labels.columns=["ecg","label1","label2","label3"]
    labels=labels.set_index("ecg")

    labels.label1=labels.label1.apply(lambda x: int(x) -1 if not np.isnan(x) else np.nan)
    labels.label2=labels.label2.apply(lambda x: int(x) -1 if not np.isnan(x) else np.nan)
    labels.label3=labels.label3.apply(lambda x: int(x) -1 if not np.isnan(x) else np.nan)
    #data
    rows=[]
    for d in tqdm(datafiles):
        sex,age,channels_available,filename_out,filename_out_relative=mat_to_np(filename_in=d, target_fs=target_fs, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder)
        rows.append({"ecg":d.stem, "dataset":"ICBEB2018", "data":filename_out_relative, "age": (np.nan if np.isnan(age) else int(age)), "sex": sex.lower(), "channels_available":channels_available})
    df=pd.DataFrame(rows)
    df=df.set_index("ecg")
    #join
    df=df.join(labels)
    df=df.reset_index()
    #define actual label (label 2 and label 3 are multilabels)
    df["label"]=df["label1"]
    df["has_label2"]=~pd.isna(df["label2"])#i.e. multilabeled label 3 will only be set if label 2 is
    df["has_label3"]=~pd.isna(df["label3"])

    #age
    df["label_age"]=df.age.apply(lambda x: _age_to_categorical(x))
    #sex
    df["label_sex"]=df.sex.apply(lambda x: _sex_to_categorical(x))

    def multi_label(x):
        res = np.zeros(len(label_itos),dtype=np.float32)
        for xi in x:
            if(np.isfinite(xi) and int(xi) in list(range(len(label_itos)))):
                res[int(xi)]=1
        return res

    def combine_labels(x):
        res = [x["label1"]]
        if(np.isfinite(x["label2"])):
            res += [int(x["label2"])]
        if(np.isfinite(x["label3"])):
            res += [int(x["label3"])]
        return res

    df["labels"]=df.apply(lambda x: combine_labels(x),axis=1)

    df["label_multi"]=df.labels.apply(lambda x:multi_label(x))
    return df, label_itos

def prepare_data_icbeb(data_folder, target_folder=None, channel_stoi=channel_stoi_canonical, channels=12, target_fs=50, strat_folds=10, cv_random_state=42,recreate_data=False,discard_valtest=True):
    target_root_icbeb = Path(".") if target_folder is None else target_folder
    if(recreate_data):
        #collect files
        datafiles_icbeb=list(data_folder.glob('**/*.mat'))
        labelfile_icbeb=data_folder/"REFERENCE.csv"

        #reformat data
        df_icbeb, lbl_itos_icbeb = reformat_data_icbeb(datafiles_icbeb,labelfile_icbeb,channel_stoi=channel_stoi,target_fs=target_fs,channels=channels,target_folder=target_folder)

        #TODO eventually move all the stuff below into reformat data
        #remove valid data
        df_icbeb["set"] = df_icbeb.data.apply(lambda x: (0 if "train" in x.parts else 1))
        if(discard_valtest is True):
            df_icbeb=df_icbeb[df_icbeb["set"]==0].reset_index()#only train set has label
        else:#reset labels that have been set erroneously
            df_icbeb.loc[df_icbeb["set"]==1,"label"]=np.nan
            df_icbeb.loc[df_icbeb["set"]==1,"label_multi"]=np.nan
            df_icbeb.loc[df_icbeb["set"]==1,"label1"]=np.nan
            df_icbeb.loc[df_icbeb["set"]==1,"label2"]=np.nan
            df_icbeb.loc[df_icbeb["set"]==1,"label3"]=np.nan

        #train test split (all)
        stratified_ids = stratify(list(df_icbeb["label_multi"]), range(len(lbl_itos_icbeb)), [1./strat_folds]*strat_folds)
        df_icbeb["strat_fold"]=-1
        idxs = np.array(df_icbeb.index.values)
        for i,split in enumerate(stratified_ids):
            df_icbeb.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df_icbeb,data_folder=target_folder)
        dataset_add_std_col(df_icbeb,data_folder=target_folder)
        dataset_add_length_col(df_icbeb,data_folder=target_folder)
        #dataset_add_median_col(df_icbeb,data_folder=target_folder)
        #dataset_add_iqr_col(df_icbeb,data_folder=target_folder)

        #save means and stds
        mean_icbeb, std_icbeb = dataset_get_stats(df_icbeb)

        #save
        save_dataset(df_icbeb, lbl_itos_icbeb, mean_icbeb, std_icbeb, target_root_icbeb)
    else:
        df_icbeb, lbl_itos_icbeb, mean_icbeb, std_icbeb = load_dataset(target_root_icbeb,df_mapped=False)
    return df_icbeb,lbl_itos_icbeb, mean_icbeb, std_icbeb


def prepare_data_ptb_xl(data_path, min_cnt=10, target_fs=100, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root_ptb_xl = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root_ptb_xl.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        ptb_xl_csv = data_path/"ptbxl_database.csv"
        df_ptb_xl=pd.read_csv(ptb_xl_csv,index_col="ecg_id")
        #print(df_ptb_xl.columns)
        df_ptb_xl.scp_codes=df_ptb_xl.scp_codes.apply(lambda x: eval(x.replace("nan","np.nan")))

        # preparing labels
        ptb_xl_label_df = pd.read_csv(data_path/"scp_statements.csv")
        ptb_xl_label_df=ptb_xl_label_df.set_index(ptb_xl_label_df.columns[0])

        ptb_xl_label_diag= ptb_xl_label_df[ptb_xl_label_df.diagnostic >0]
        ptb_xl_label_form= ptb_xl_label_df[ptb_xl_label_df.form >0]
        ptb_xl_label_rhythm= ptb_xl_label_df[ptb_xl_label_df.rhythm >0]

        diag_class_mapping={}
        diag_subclass_mapping={}
        for id,row in ptb_xl_label_diag.iterrows():
            if(isinstance(row["diagnostic_class"],str)):
                diag_class_mapping[id]=row["diagnostic_class"]
            if(isinstance(row["diagnostic_subclass"],str)):
                diag_subclass_mapping[id]=row["diagnostic_subclass"]

        df_ptb_xl["label_all"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys()])
        df_ptb_xl["label_diag"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_diag.index])
        df_ptb_xl["label_form"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_form.index])
        df_ptb_xl["label_rhythm"]= df_ptb_xl.scp_codes.apply(lambda x: [y for y in x.keys() if y in ptb_xl_label_rhythm.index])

        df_ptb_xl["label_diag_subclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_subclass_mapping[y] for y in x if y in diag_subclass_mapping])
        df_ptb_xl["label_diag_superclass"]= df_ptb_xl.label_diag.apply(lambda x: [diag_class_mapping[y] for y in x if y in diag_class_mapping])

        df_ptb_xl["dataset"]="ptb_xl"
        #filter and map (can be reapplied at any time)
        df_ptb_xl, lbl_itos_ptb_xl =map_and_filter_labels(df_ptb_xl,min_cnt=min_cnt,lbl_cols=["label_all","label_diag","label_form","label_rhythm","label_diag_subclass","label_diag_superclass"])

        filenames = []
        for id, row in tqdm(list(df_ptb_xl.iterrows())):
            # always start from 500Hz and sample down
            filename = data_path/row["filename_hr"] #data_path/row["filename_lr"] if target_fs<=100 else data_path/row["filename_hr"]
            sigbufs, header = wfdb.rdsamp(str(filename))
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root_ptb_xl/(filename.stem+".npy"),data)
            filenames.append(Path(filename.stem+".npy"))
        df_ptb_xl["data"] = filenames

        #add means and std
        dataset_add_mean_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_std_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        dataset_add_length_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean_ptb_xl, std_ptb_xl = dataset_get_stats(df_ptb_xl)

        #save
        save_dataset(df_ptb_xl,lbl_itos_ptb_xl,mean_ptb_xl,std_ptb_xl,target_root_ptb_xl)
    else:
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = load_dataset(target_root_ptb_xl,df_mapped=False)
    return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl

def map_and_filter_labels(df,min_cnt,lbl_cols):
    #filter labels
    def select_labels(labels, min_cnt=10):
        lbl, cnt = np.unique([item for sublist in list(labels) for item in sublist], return_counts=True)
        return list(lbl[np.where(cnt>=min_cnt)[0]])
    df_ptb_xl = df.copy()
    lbl_itos_ptb_xl = {}
    for selection in lbl_cols:
        if(min_cnt>0):
            label_selected = select_labels(df_ptb_xl[selection],min_cnt=min_cnt)
            df_ptb_xl[selection+"_filtered"]=df_ptb_xl[selection].apply(lambda x:[y for y in x if y in label_selected])
            lbl_itos_ptb_xl[selection+"_filtered"] = np.array(sorted(list(set([x for sublist in df_ptb_xl[selection+"_filtered"] for x in sublist]))))
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection+"_filtered"])}
            df_ptb_xl[selection+"_filtered_numeric"]=df_ptb_xl[selection+"_filtered"].apply(lambda x:[lbl_stoi[y] for y in x])
        #also lbl_itos and ..._numeric col for original label column
        lbl_itos_ptb_xl[selection]= np.array(sorted(list(set([x for sublist in df_ptb_xl[selection] for x in sublist]))))
        lbl_stoi = {s:i for i,s in enumerate(lbl_itos_ptb_xl[selection])}
        df_ptb_xl[selection+"_numeric"]=df_ptb_xl[selection].apply(lambda x:[lbl_stoi[y] for y in x])
    return df_ptb_xl, lbl_itos_ptb_xl

def prepare_data_sph(data_path, min_cnt=10, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    target_root_sph = Path(".") if target_folder is None else target_folder
    target_root_sph.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        # reading df
        df_sph = pd.read_csv(data_path/'metadata.csv')
        df_sph.columns = ["ecg_id","aha_code","patient_id","age","sex","n","date"]
        df_sph = df_sph.drop("n",axis=1)
        
        #human readable labels
        #description = pd.read_csv(data_path/'code.csv')#descriptions not used here
        #def aha_dict(desc):
        #    res={}
        #    for _,row in desc.iterrows():
        #        res[str(row["Code"])]=row["Description"]
        #    return res
        #def aha_to_text(t):
        #    tsplit=[x.split("+") for x in t.split(";")]
        #    print(tsplit)
        #    return ";".join(["+".join([ahad[y] for y in x]) for x in tsplit])

        # preparing labels
        df_sph["label_primary_w_mod"]=df_sph.aha_code.apply(lambda x: x.split(";"))
        df_sph["label_primary"]= df_sph.label_primary_w_mod.apply(lambda x: [y.split("+")[0] for y in x])

        df_sph["dataset"]="sph"
        #filter and map (can be reapplied at any time)
        df_sph, lbl_itos_sph =map_and_filter_labels(df_sph,min_cnt=min_cnt,lbl_cols=["label_primary","label_primary_w_mod"])

        channel_labels = ["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
        channel_itos=   ["x"]*channels
        for k in channel_stoi.keys():
            if(channel_stoi[k]<channels):
                channel_itos[channel_stoi[k]]=k
        chidxs=np.array([np.where(np.array(channel_labels)==c.lower())[0][0] for c in np.array(channel_itos)[:channels]])

        filenames = []
        for index in tqdm(range(len(df_sph))):
            record = df_sph.ecg_id[index]
            filename = data_path/"records"/(str(record)+".h5")
            with h5py.File(filename, 'r') as f:
                signal = f['ecg'][()].astype('float32') # K,L
                resampled_signal = resampy.resample(signal, 500, target_fs, axis=1)
                ordered_signal = resampled_signal[chidxs].T #L,K
                np.save(target_root_sph/(filename.stem+".npy"),ordered_signal)
            filenames.append(Path(filename.stem+".npy"))
        df_sph["data"] = filenames

        #stratified splits
        lbl_itos = lbl_itos_sph["label_primary_w_mod"]
        df_labeled = df_sph.copy()
        df_labeled["label_sex"] = df_labeled.sex.apply(lambda x: len(lbl_itos)+1 if x=="M" else len(lbl_itos)+2)
        df_labeled["label_age"] = df_labeled.age.apply(lambda x: len(lbl_itos)+3 if x<20 else (len(lbl_itos)+4 if x<40 else (len(lbl_itos)+5 if x<60 else (len(lbl_itos)+6 if x<80 else len(lbl_itos)+7 ))))

        df_labeled["labelx"] = df_labeled.apply(lambda row: row["label_primary_w_mod_numeric"]+[len(lbl_itos)]+[row["label_sex"]]+[row["label_age"]],axis=1)

        df_patients = df_labeled.groupby("patient_id")["labelx"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify(patients_labels, range(len(lbl_itos)+8), [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True)
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        df_sph["strat_fold"]=-1 #unlabeled will end up in fold -1
        for i,split in enumerate(stratified_patient_ids):
            df_sph.loc[df_sph.patient_id.isin(split),"strat_fold"]=i

        #original splits
        # 80%-20% split
        # put all records belonging to patients with
        # multiple records in the test set
        test1 = df_sph.patient_id.duplicated(keep=False)
        N = int(len(df_sph)*0.2) - sum(test1)
        # 73 is chosen such that all primary statements exist in both sets
        df_test = pd.concat([df_sph[test1], df_sph[~test1].sample(N, random_state=73)])
        df_sph["orig_fold"]=df_sph.ecg_id.apply(lambda x: int(x in np.array(df_test.ecg_id)))
        
        #add means and std
        dataset_add_mean_col(df_sph,data_folder=target_root_sph)
        dataset_add_std_col(df_sph,data_folder=target_root_sph)
        dataset_add_length_col(df_sph,data_folder=target_root_sph)
        
        #save means and stds
        mean_sph, std_sph = dataset_get_stats(df_sph)

        #save
        save_dataset(df_sph,lbl_itos_sph,mean_sph,std_sph,target_root_sph)
    else:
        df_sph, lbl_itos_sph, mean_sph, std_sph = load_dataset(target_root_sph,df_mapped=False)
    return df_sph, lbl_itos_sph, mean_sph, std_sph



def thew_to_np(filename_in, target_fs=100, channels=12, max_length_seconds=0, channel_stoi=None, fs=180, target_folder=None):
    x = Holter(str(filename_in))
    metadata = {"id":x.id, "length":x.ecg_size, "record_time": str(x.record_date)+" "+str(x.start_time), "channel_labels":[xx.spec_str().lower() for xx in x.lead]}
    metadata["channels_available"] = get_available_channels(metadata["channel_labels"],channel_stoi)

    x.load_data()
    sigbufs = np.transpose(np.array([l.data for l in x.lead]))#timesteps,channel

    if(max_length_seconds>0):
        max_length = min(len(sigbufs),int(np.round(fs*max_length_seconds)))
        sigbufs = sigbufs[:max_length]

    data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=metadata["channel_labels"],fs=fs,target_fs=target_fs, channels=channels)

    filename_out, filename_out_relative = get_filename_out(filename_in,target_folder)
    np.save(filename_out, data.astype(np.float32))
    metadata["length"]=len(data)

    return metadata, filename_out, filename_out_relative


def reformat_data_thew(data_path=".",max_length_seconds=0, target_fs=100, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None):
    data_path = Path(data_path)

    #prepare clinical data
    df_clinical = pd.read_excel(data_path/"clinicalData"/"E-HOL-12-1172-012.xls",sheet_name=1)
    df_clinical = df_clinical[~df_clinical["ECG file"].isna()]#filter out nan entries
    label_itos=["Non ACS","Non-cardiac", "Unstb Angina","Non-ST Elev MI","ST-Elev MI"]#,"nan"]
    label_stoi={s:i for i,s in enumerate(label_itos)}
    df_clinical["label"]=df_clinical.FinalDx.apply(lambda x: label_stoi[x])
    df_clinical["label_txt"]=df_clinical.FinalDx
    df_clinical["dataset"]="THEW Chest Pain LR"
    df_clinical = df_clinical[~df_clinical.StudyID.isna()]
    df_clinical = df_clinical.astype({'StudyID': 'int32'})
    df_clinical = df_clinical.set_index("StudyID")


    res = []
    for file in tqdm(list(data_path.glob('**/*.ecg'))):
        metadata, filename_out, filename_out_relative = thew_to_np(file, target_fs=target_fs, channels=channels, max_length_seconds=max_length_seconds, channel_stoi=channel_stoi, fs=180, target_folder=target_folder)
        res.append({"ecg_id":file.stem.replace("S-","").replace("S","").lstrip(), "patient_id":int(file.stem.replace("S-","").replace("S","").lstrip()[:4]), "data":filename_out_relative, "length": metadata["length"], "channels_available": metadata["channels_available"]})
    df = pd.DataFrame(res)

    df = df.join(df_clinical, how="left", on="patient_id")
    df_error= list(df[df["ECG file"].isna()].ecg_id)
    if(len(df_error)>0):
        print("Skipping the following ECG files that could not be matched to diagnoses:",df_error)
        df = df[~df["ECG file"].isna()]
    return df, label_itos


def prepare_data_thew(data_folder, max_length_seconds=0, target_folder=None, channel_stoi=channel_stoi_canonical, channels=12, target_fs=100, strat_folds=10, cv_random_state=42,recreate_data=False):
    target_root_thew = Path(".") if target_folder is None else target_folder
    if(recreate_data):
        #reformat data
        df_thew, lbl_itos_thew = reformat_data_thew(data_path=data_folder,max_length_seconds=max_length_seconds,channel_stoi=channel_stoi,target_fs=target_fs,channels=channels,target_folder=target_folder)

        #cv splits
        df_patients = df_thew.groupby("patient_id")["label"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients)
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify(patients_labels, range(6), [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs)
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        df_thew["strat_fold"]=-1
        for i,split in enumerate(stratified_patient_ids):
            df_thew.loc[df_thew.patient_id.isin(split),"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df_thew,data_folder=target_root_thew)
        dataset_add_std_col(df_thew,data_folder=target_root_thew)
        dataset_add_length_col(df_thew,data_folder=target_root_thew)
        #dataset_add_median_col(df_thew,data_folder=target_root_thew)
        #dataset_add_iqr_col(df_thew,data_folder=target_root_thew)

        #save means and stds
        mean_thew, std_thew = dataset_get_stats(df_thew)

        #save
        save_dataset(df_thew, lbl_itos_thew, mean_thew, std_thew, target_root_thew)
    else:
        df_thew, lbl_itos_thew, mean_thew, std_thew = load_dataset(target_root_thew,df_mapped=False)
    return df_thew,lbl_itos_thew, mean_thew, std_thew


def prepare_data_cinc(data_path, min_cnt=10, datasets=None, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''unzip archives into separate folders with dataset names from above
    use datasets e.g. ["ICBEB2018","ICBEB2018_2","INCART","PTB","PTB-XL","Georgia"] to filter particular datasets (None for all)
    '''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        dx_meta = pd.concat([pd.read_csv(data_path/"dx_mapping_scored.csv"),pd.read_csv(data_path/"dx_mapping_unscored.csv")],sort=True)
        if("SNOMEDCTCode" in dx_meta.columns):
            dx_meta = dx_meta.rename(columns={"SNOMEDCTCode":"SNOMED CT Code"})
        print(dx_meta.columns)  
        dx_mapping_snomed_abbrev = {a:b for [a,b] in list(dx_meta.apply(lambda row: [row["SNOMED CT Code"],row["Abbreviation"]],axis=1))}

        metadata = []
        for filename in tqdm(list(data_path.glob('**/*.hea'))):
            if(datasets is not None and not(filename.parts[-2] in datasets)):
                continue
            try:
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            except:
                print("Warning:",str(filename),"is corrupt. Skipping.")
                continue
            #print(filename,sigbufs.shape,np.min(sigbufs,axis=0),np.any(np.isnan(sigbufs)))
            if(np.any(np.isnan(sigbufs))):
                print("Warning:",str(filename)[:-4]," data contains nans. Skipping.")
                continue
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            assert(target_fs<=header['fs'])
            np.save(target_root/(filename.stem+".npy"),data)
            labels=[]
            age=np.nan
            sex="nan"
            for l in header["comments"]:
                arrs = l.strip().split(' ')
                if l.startswith('Dx:'):
                    labels = [dx_mapping_snomed_abbrev[int(x)] for x in arrs[1].split(',')]
                elif l.startswith('Age:'):
                    try:
                        age = int(arrs[1])
                    except:
                        age= np.nan
                elif l.startswith('Sex:'):
                    sex = arrs[1].strip().lower()
                    if(sex=="m"):
                        sex="male"
                    elif(sex=="f"):
                        sex="female"

            metadata.append({"data":Path(filename.stem+".npy"),"label":labels,"sex":sex,"age":age,"dataset":"cinc_"+filename.parts[-2]})
        df =pd.DataFrame(metadata)

        #lbl_itos = np.unique([item for sublist in list(df.label) for item in sublist])
        #lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
        #df["label"] = df["label"].apply(lambda x: [lbl_stoi[y] for y in x])

        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label"])

        #does not incorporate patient-level split
        df["strat_fold"]=-1
        for ds in np.unique(df["dataset"]):
            print("Creating CV folds:",ds)
            dfx = df[df.dataset==ds]
            idxs = np.array(dfx.index.values)
            lbl_itosx = np.unique([item for sublist in list(dfx.label) for item in sublist])
            stratified_ids = stratify(list(dfx["label"]), lbl_itosx, [1./strat_folds]*strat_folds)

            for i,split in enumerate(stratified_ids):
                df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


def prepare_data_chapman(data_path, min_cnt=10, denoised=False, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''prepares the Chapman dataset from Zheng et al 2020'''
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        #df_attributes = pd.read_excel("./AttributesDictionary.xlsx")
        #df_conditions = pd.read_excel("./ConditionNames.xlsx")
        #df_rhythm = pd.read_excel("./RhythmNames.xlsx")
        df = pd.read_excel(data_path/"Diagnostics.xlsx")
        df["id"]=df.FileName
        df["data"]=df.FileName.apply(lambda x: x+".npy")
        df["label_condition"]=df.Beat.apply(lambda x: [y for y in x.split(" ") if x!="NONE"])
        df["label_rhythm"]=df.Rhythm.apply(lambda x: x.split(" "))
        df["label_all"]=df.apply(lambda row: row["label_condition"]+row["label_rhythm"],axis=1)
        df["sex"]=df.Gender.apply(lambda x:x.lower())
        df["age"]=df.PatientAge
        df.drop(["Gender","PatientAge","Rhythm","Beat","FileName"],inplace=True,axis=1)

        #map to numerical indices
        #lbl_itos={}
        #lbl_stoi={}
        #lbl_itos["all"] = np.unique([item for sublist in list(df.label_txt) for item in sublist])
        #lbl_stoi["all"] = {s:i for i,s in enumerate(lbl_itos["all"])}
        #df["label"] = df["label_txt"].apply(lambda x: [lbl_stoi["all"][y] for y in x])
        #lbl_itos["condition"] = np.unique([item for sublist in list(df.label_condition_txt) for item in sublist])
        #lbl_stoi["condition"] = {s:i for i,s in enumerate(lbl_itos["condition"])}
        #df["label_condition"] = df["label_condition_txt"].apply(lambda x: [lbl_stoi["condition"][y] for y in x])
        #lbl_itos["rhythm"] = np.unique([item for sublist in list(df.label_rhythm_txt) for item in sublist])
        #lbl_stoi["rhythm"] = {s:i for i,s in enumerate(lbl_itos["rhythm"])}
        #df["label_rhythm"] = df["label_rhythm_txt"].apply(lambda x: [lbl_stoi["rhythm"][y] for y in x])
        
        #filter (can be reapplied at any time)
        df, lbl_itos =map_and_filter_labels(df,min_cnt=min_cnt,lbl_cols=["label_all","label_condition","label_rhythm"])
        
        df["dataset"]="Chapman"

        for id,row in tqdm(list(df.iterrows())):
            fs = 500.

            df_tmp = pd.read_csv(data_path/("ECGDataDenoised" if denoised else "ECGData")/(row["id"]+".csv"))
            channel_labels = list(df_tmp.columns)
            sigbufs = np.array(df_tmp)*0.001 #assuming data is given in muV

            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels)
            assert(target_fs<=fs)
            np.save(target_root/(row["id"]+".npy"),data)

        stratified_ids = stratify(list(df["label_all"]), lbl_itos["label_all"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


def prepare_data_ribeiro_test(data_path, denoised=False, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''prepares test set of Ribeiro et al Nat Comm 2020'''
    data_path = Path(data_path)
    target_root = Path(".") if target_folder is None else target_folder
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        lbl_itos = ["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]
        channel_labels = ["i","ii","iii","avr","avl","avf","v1","v2","v3","v4","v5","v6"]
        fs= 400.
        #prepare df
        df_cardiologist1 = pd.read_csv(data_path/"annotations"/"cardiologist1.csv")
        df_cardiologist2 = pd.read_csv(data_path/"annotations"/"cardiologist2.csv")
        df_gold = pd.read_csv(data_path/"annotations"/"gold_standard.csv")
        df_cardiology_residents = pd.read_csv(data_path/"annotations"/"cardiology_residents.csv")
        df_dnn = pd.read_csv(data_path/"annotations"/"dnn.csv")
        df_emergency_residents = pd.read_csv(data_path/"annotations"/"emergency_residents.csv")
        df_medical_students = pd.read_csv(data_path/"annotations"/"medical_students.csv")
        df_attributes = pd.read_csv(data_path/"attributes.csv")
        sex_map = {"M":"male", "F":"female"}
        df_attributes.sex = df_attributes.sex.apply(lambda x: sex_map[x])

        def reformat_predictions(df, colname_txt="label_txt", colname_num="label", lbl_itos=["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]):
            lbl_stoi = {s:i for i,s in enumerate(lbl_itos)}
            df[colname_txt]=df.apply(lambda row: ("1AVB " if row["1dAVb"] else "")+("RBBB " if row["RBBB"] else "")+("LBBB " if row["LBBB"] else "")+("SBRAD " if row["SB"] else "")+("AFIB " if row["AF"] else "")+("STACH " if row["ST"] else ""),axis=1)
            df[colname_txt]=df[colname_txt].apply(lambda x: [y for y in x.strip().split(" ") if y!=""])
            df[colname_num]=df[colname_txt].apply(lambda x: [lbl_stoi[y] for y in x if y in lbl_stoi.keys()])
            df.drop(["1dAVb","RBBB","LBBB","SB","AF","ST"],axis=1,inplace=True)

        reformat_predictions(df_cardiologist1,"label_cardiologist1_txt","label_cardiologist1")
        reformat_predictions(df_cardiologist2,"label_cardiologist2_txt","label_cardiologist2")
        reformat_predictions(df_gold,"label_txt","label")
        reformat_predictions(df_cardiology_residents,"label_cardiology_residents_txt","label_cardiology_residents")
        reformat_predictions(df_emergency_residents,"label_emergency_residents_txt","label_emergency_residents")
        reformat_predictions(df_medical_students,"label_medical_students_txt","label_medical_students")
        reformat_predictions(df_dnn,"label_dnn_txt","label_dnn")

        df=df_gold.join([df_cardiologist1,df_cardiologist2,df_cardiology_residents,df_emergency_residents,df_medical_students,df_dnn,df_attributes])
        df["data"]=[Path("code_test_"+str(i)+".npy") for i in range(len(df))]
        df["dataset"]="code_test"
        #prepare raw data
        with h5py.File(data_path/"ecg_tracings.hdf5", "r") as f:
            sigbufs = np.array(f['tracings'])
        start_idxs=[ np.where(np.sum(np.abs(sigbufs[i]),axis=1)==0.)[0] for i in range(len(sigbufs))] #discard zeros at beginning/end
        start_idxs = [len(a)//2 for a in start_idxs]


        for id,row in tqdm(list(df.iterrows())):
            data = resample_data(sigbufs=sigbufs[id][start_idxs[id]:-start_idxs[id] or None],channel_stoi=channel_stoi,channel_labels=channel_labels,fs=fs,target_fs=target_fs,channels=channels)
            assert(target_fs<=fs)
            np.save(target_root/(row["data"]),data)

        stratified_ids = stratify(list(df.apply(lambda row: row["label_txt"]+[row["sex"]],axis=1)), lbl_itos+["male","female"], [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


def prepare_data_ludb(data_path, subdivide=True, extra_segments=True, crop_bg=True, min_cnt=10, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''subdivide: divide p,qrs=N,t into two parts; extra_segments: annotate also pr,st and tp segments crop_bg:remove bg only at start and end'''
    target_root = Path(".") if target_folder is None else target_folder
    #print(target_root_ptb_xl)
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        def parse_comments(comments):
            age = np.nan
            sex = "nan"
            diag = []
            diag_flag=False
            axis = "nan"
            for x in comments:
                if(x.startswith("<age>")):
                    age = int(x.split(":")[1].strip().replace(">89","90"))
                elif(x.startswith("<sex>")):
                    sex = "male" if x.split(":")[1].strip() == "M" else "female"
                elif(x.startswith("Electric axis of the heart")):
                    axis = x.split(":")[1].strip()[:-1]
                elif(x.startswith("<diagnoses>")):
                    diag_flag=True
                else:
                    if(diag_flag):
                        diag.append(x[:-1].lower())
            return {"age":age,"sex":sex,"axis":axis,"diag":diag}

        def convert_ann(sample,symbol,length=5000,target_fs=100,fs=500,subdivide=True,extra_segments=True):

            x=np.zeros(int(np.round(target_fs/fs*length)),dtype=np.int8)

            label_itos = ["BG","(N","(t","(p","N)","t)","p)"] if subdivide else ["BG","N","t","p"]
            if(extra_segments):
                label_itos += ["pr","st","tp"]
            label_stoi = {s:i for i,s in enumerate(label_itos)}
            i=0
            while(i<len(symbol)-1):
                #incomplete first segment
                if(subdivide and i==0 and (symbol[i] in ["N","t","p"]) and symbol[i+1]==")"):
                    mid = int(np.round(target_fs/fs*sample[i]))
                    end = int(np.round(target_fs/fs*sample[i+1]))
                    label = label_stoi[symbol[i]+")"]
                    x[mid:end]=label
                    i+=2
                #incomplete last segment
                elif(subdivide and i==len(symbol)-2 and symbol[i]=="(" and symbol[i+1] in ["N","t","p"]):
                    start = int(np.round(target_fs/fs*sample[i]))
                    mid = int(np.round(target_fs/fs*sample[i+1]))
                    label = label_stoi["("+symbol[i+1]]
                    x[start:mid]=label
                    i+=2
                #only take into account complete beats ( Ntp )
                elif(i<len(symbol)-2 and symbol[i]=="(" and symbol[i+1] in ["N","t","p"] and symbol[i+2]==")"):
                    start = int(np.round(target_fs/fs*sample[i]))
                    end = int(np.round(target_fs/fs*sample[i+2]))
                    if(subdivide):
                        mid = int(np.round(target_fs/fs*sample[i+1]))
                        label1 = label_stoi["("+symbol[i+1]]
                        label2 = label_stoi[symbol[i+1]+")"]
                        x[start:mid]=label1
                        x[mid:end]=label2
                    else:
                        label = label_stoi[symbol[i+1]]
                        x[start:end]=label
                    i=i+3
                else:
                    i=i+1

            if(extra_segments):
                def ranges(nums):
                    nums = sorted(set(nums))
                    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
                    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
                    return list(zip(edges, edges))
                bgs = np.where(np.array(x)==0)[0]
                bgs_ranges = ranges(bgs)

                for start,end in bgs_ranges:
                    if(start>0 and end<len(x)-1):
                        if(subdivide and x[start-1]==label_stoi["p)"] and x[end+1]==label_stoi["(N"]):
                            x[start:end+1]=label_stoi["pr"]
                        elif(not(subdivide) and x[start-1]==label_stoi["p"] and x[end+1]==label_stoi["N"]):
                            x[start:end+1]=label_stoi["pr"]
                        elif(subdivide and x[start-1]==label_stoi["N)"] and x[end+1]==label_stoi["(t"]):
                            x[start:end+1]=label_stoi["st"]
                        elif(not(subdivide) and x[start-1]==label_stoi["N"] and x[end+1]==label_stoi["t"]):
                            x[start:end+1]=label_stoi["st"]
                        elif(subdivide and x[start-1]==label_stoi["t)"] and x[end+1]==label_stoi["(p"]):
                            x[start:end+1]=label_stoi["tp"]
                        elif(not(subdivide) and x[start-1]==label_stoi["t"] and x[end+1]==label_stoi["p"]):
                            x[start:end+1]=label_stoi["tp"]



            return x, label_itos

        metadata = []
        lbl_itos_ann = []
        for filename in tqdm(list(data_path.glob('**/*.hea'))):

            sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            meta = parse_comments(header["comments"])
            #data
            data = data * 0.001 #fix mV
            assert(target_fs<=header['fs'])
            meta["data"]= Path(filename.stem+".npy")

            #read and convert annotations
            anns=[wfdb.rdann(str(filename)[:-4],"atr_"+s) for s in header["sig_name"]]
            data_ann = []
            for ann in anns:
                ann_conv, lbl_itos_ann = convert_ann(ann.sample,ann.symbol,len(sigbufs),target_fs,header["fs"],subdivide=subdivide,extra_segments=extra_segments)
                data_ann.append(ann_conv)
            data_ann = np.array(data_ann).transpose()
            meta["annotation"] = Path(filename.stem+"_ann.npy")

            if(crop_bg):#remove bg only parts
                bgonly_idxs=np.where(np.all(data_ann==0,axis=1))[0]
                for start in range(len(data)):
                    if(not(start in bgonly_idxs)):
                        break
                for end in range(len(data)-1,-1,-1):
                    if(not(end in bgonly_idxs)):
                        break
                data=data[start:end+1]
                data_ann=data_ann[start:end+1]
            np.save(target_root/(filename.stem+".npy"),data)
            np.save(target_root/(filename.stem+"_ann.npy"),data_ann)

            metadata.append(meta)

        df =pd.DataFrame(metadata)
        df["dataset"]="ludb"
        #global label
        diag_list = [item for sublist in list(df.diag) for item in sublist]
        a,b = np.unique(diag_list,return_counts=True)
        idxs=np.where(b>=10)
        lbl_itos_global = a[idxs]
        lbl_stoi_global = {s:i for i,s in enumerate(lbl_itos_global)}
        df["label_global"] = df.diag.apply(lambda x:[lbl_stoi_global[y] for y in x if y in lbl_stoi_global.keys()])
        df["label_global_txt"] = df.diag.apply(lambda x:[y for y in x if y in lbl_stoi_global.keys()])

        lbl_itos={"global":lbl_itos_global, "annotation":lbl_itos_ann}

        stratified_ids = stratify(list(df["label_global_txt"]), lbl_itos_global, [1./strat_folds]*strat_folds)
        df["strat_fold"]=-1
        idxs = np.array(df.index.values)
        for i,split in enumerate(stratified_ids):
            df.loc[idxs[split],"strat_fold"]=i

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


def prepare_data_ribeiro_full(data_path, code15=False, max_records_per_id_exam = 0, min_records_per_id_exam=0, overwrite_npy_if_exists=False, selected_folds=None, skip_strat_folds=False, subsample=None, target_fs=100, strat_folds=100, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    data_path = Path(data_path)
    target_root = Path(".") if target_folder is None else Path(target_folder)
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):

        #1 prepare filenames df
        print("Loading filenames...")
        if(code15):
            res=[]
            for f in data_path.glob('*.hdf5'):
                f5 = h5py.File(f, 'r')
                for i,x in enumerate(f5['exam_id']):
                    res.append({"filename":f, "id_row":i, "id_exam":x})
            df = pd.DataFrame(res)
        else:
            with open(data_path/"wfdb/RECORDS.txt", 'r') as file:
                filenames = file.readlines()

            res=[]
            for filename in tqdm(filenames,leave=False):
                filename = filename.strip()
                if(len(filename)>0):
                    filename = data_path/"wfdb"/filename
                    filename_stem = filename.stem

                    try:
                        id_exam = int(filename_stem.split("_N")[0][4:])
                        id_record = int(filename_stem.split("_N")[1])
                    except:
                        print("Error processing",filename,". Skipping.")
                        continue
                    res.append({"filename":filename.parent/filename.stem, "id_exam": id_exam, "id_record":id_record})
            #filename without suffix, id_exam matching annotations.csv id_record=1... (record within the same id_exam)
            df = pd.DataFrame(res)

        #2 load annotations
        print("Loading annotations...")
        if(code15):
            df_annotations=pd.read_csv(data_path/"exams.csv").set_index("exam_id")
        else:
            df_annotations=pd.read_csv(data_path/"annotations.csv").set_index("id_exam")

        #3 minor postprocessing
        df = df.join(df_annotations, how="left", on="id_exam")
        df["dataset"]="code15" if code15 else "code100"

        if(code15 is False):
            df["date_exam"]=pd.to_datetime(df["date_exam"])
        df["missing_label"] = df.ST.isna()
        if(code15):
            df["sex"] =df.is_male.apply(lambda x: "male" if x else "female")
        else:
            sex_map = {"M":"male", "F":"female"}
            df["sex"] = df.sex.apply(lambda x: sex_map[x] if x in sex_map.keys() else "unknown").astype("category")


        df["label"]=df[["1dAVb","RBBB","LBBB","SB","AF","ST"]].values.tolist()
        df["label"]=df.label.apply(lambda x: list(np.where(np.array(x)==1.)[0]))
        lbl_itos = ["1AVB","RBBB","LBBB","SBRAD","AFIB","STACH"]
        if(code15):
            df.rename({'patient_id': 'id_patient'}, axis=1, inplace=True)
        print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients before filtering.")

        #4 filter if desired
        if(not code15 and (max_records_per_id_exam > 0 or min_records_per_id_exam > 0)):
            if(max_records_per_id_exam > 0):
                df = df[(df.id_record<=max_records_per_id_exam) & (df.id_record>=min_records_per_id_exam)].copy()
            else:
                df = df[df.id_record>=min_records_per_id_exam].copy()
            print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after filtering for max_records_per_id_exam.")

        #subsample based on patients if desired
        if(subsample is not None):
            df_labeled = df[df.missing_label==False].copy()
            patient_ids = np.array(df.id_patient.unique())
            patient_ids_selected = patient_ids[int(subsample[0]*len(patient_ids)):int(subsample[1]*len(patient_ids))]
            df_labeled = df_labeled[df_labeled.id_patient.isin(patient_ids_selected)]


            df_unlabeled = df[df.missing_label==True].copy()
            df_unlabeled=df_unlabeled.iloc[int(subsample[0]*len(df_unlabeled)):int(subsample[1]*len(df_unlabeled))]

            df = pd.concat([df_labeled,df_unlabeled])
            print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after subsampling.")


        #5 determine folds
        if(skip_strat_folds is False):
            print("Preparing labels for fold distribution...")
            df_labeled = df[df.missing_label==False].copy()
            df_labeled["label_sex"] = df_labeled.sex.apply(lambda x: len(lbl_itos)+1 if x=="male" else len(lbl_itos)+2)
            df_labeled["label_age"] = df_labeled.age.apply(lambda x: len(lbl_itos)+3 if x<20 else (len(lbl_itos)+4 if x<40 else (len(lbl_itos)+5 if x<60 else (len(lbl_itos)+6 if x<80 else len(lbl_itos)+7 ))))

            df_labeled["labelx"] = df_labeled.apply(lambda row: row["label"]+[len(lbl_itos)]+[row["label_sex"]]+[row["label_age"]],axis=1)

            df_patients = df_labeled.groupby("id_patient")["labelx"].apply(lambda x: list(x))
            patients_ids = list(df_patients.index)
            patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
            patients_num_ecgs = list(df_patients.apply(len))

            stratified_ids = stratify_batched(patients_labels, range(len(lbl_itos)+8), [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True,batch_size=10000)
            stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

            df["strat_fold"]=-1 #unlabeled will end up in fold -1
            for i,split in enumerate(stratified_patient_ids):
                df.loc[df.id_patient.isin(split),"strat_fold"]=i
            print("Fold distribution:",df.strat_fold.value_counts())
            if(selected_folds is not None):
                df=df[df.strat_fold.isin(selected_folds)].copy()
                print("In total",len(df),"records from more than",len(df.id_patient.unique()),"patients after filtering for selected folds.")


        #6 actually prepare the files
        print("Reformating files...")
        filenames = []
        times=[]
        keeplist = []

        def process_single(i,df):
            filename = df.filename.iloc[i]
            keep = True
            if(code15):
                filename_out = "code15_"+str(df.id_exam.iloc[i])+".npy"
            else:
                filename_out = "code100_"+filename.stem+".npy"
            if(overwrite_npy_if_exists or not((target_root/filename_out).exists())):#recreate npy
                if(code15):
                    with h5py.File(filename, 'r') as f5:
                        sigbufs = np.array(f5['tracings'][df.id_row.iloc[i]])
                    start_idxs=np.where(np.sum(np.abs(sigbufs),axis=1)==0.)[0] #discard zeros at beginning/end
                    start_idx = len(start_idxs)//2
                    sigbufs = sigbufs[start_idx:-start_idx or None]
                    sig_names = "I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6".lower().split(",")
                    fs = 400
                else:
                    sigbufs, header = wfdb.rdsamp(str(filename))#str(row["filename"]))
                    sig_names = [x.lower().replace("d","") for x in header["sig_name"]]
                    fs = header['fs']
                if(len(sigbufs)==0):
                    print("Error processing id",i,"filename",filename.stem,"empty waveform. Will be dropped.")
                    keep = False
                elif(target_fs>fs):
                    print("Error processing id",i,"filename",filename.stem,"sampling frequency insufficient. Will be dropped.")
                    keep = False
                else:
                    data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=sig_names,fs=fs,target_fs=target_fs,channels=channels)
                    np.save(target_root/filename_out,data)
            else:# use existing file
                if(code15 is False):
                    record = wfdb.rdheader(str(filename))
                    header = {"base_date":record.base_date, "base_time":record.base_time}
            if(code15):
                time = pd.NaT
            else:    
                if(header["base_date"] is not None and header["base_time"] is not None):
                    time = datetime.datetime.combine(header["base_date"],header["base_time"])
                elif(header["base_date"] is not None):
                    time = header["base_date"]
                else:
                    time = pd.NaT
            return i, filename_out, time, keep

        #for id, row in tqdm(df.iterrows()):
        for i in tqdm(range(len(df))):
            filename = df.filename.iloc[i]
            _,filename_out,time,keep = process_single(i,df)
            times.append(time)
            filenames.append(filename_out)
            keeplist.append(keep)

        ####
        #import multiprocessing
        #from joblib import Parallel, delayed


        #num_cores = multiprocessing.cpu_count()
        #inputs = tqdm(range(len(df)))

        #processed_list = Parallel(n_jobs=num_cores)(delayed(process_single)(i,df) for i in inputs)
        #ids, filenames, times = zip(*processed_list)
        #df_result = pd.DataFrame({"id":ids,"data":filenames,"record_datetime":times}).set_index("id")
        #df["id"]= range(len(df))
        #df = df.join(df_result,on="id").drop("id",axis=1)


        df["data"] = filenames
        df["record_datetime"]=times
        df["keep"]=keeplist

        df= df[df["keep"]].copy()
        df.drop(["keep"],axis=1,inplace=True)

        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)
        #dataset_add_median_col(df,data_folder=target_root)
        #dataset_add_iqr_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df,lbl_itos,mean,std,target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

##############################################################################################################
# READING DATASETS WITH PHYSIONET ANNOTATION FILES
##############################################################################################################

###########################################
#UTILITY FUNCTIONS
###########################################

def create_segmentation_mask(sig_len,ann_sample,ann_annotation,ann_stoi,ann_unk_id,discard_labels=[""]):
    '''
    sig_len: length of the full sequence
    ann_sample: timestamps of annotations (in pixel)
    ann_annotation: corresponding annotations
    ann_stoi: dict which maps annotation symbols to integers
    ann_unk_id: id to which annotations are mapped by default
    discard_labels: incoming annotation types to be discarded (typically at least empty string)
    '''
    ann_unk_id
    mask = np.zeros(sig_len,dtype=np.int8)+ann_unk_id
    
    ann_annotation = [x.replace('\x00','') for x in ann_annotation]
    ann_sample = np.array(ann_sample)[np.where(np.isin(ann_annotation,discard_labels)==False)[0]]
    ann_annotation = np.array(ann_annotation)[np.where(np.isin(ann_annotation,discard_labels)==False)[0]]
    
    for sym,sta,end in zip(ann_annotation,ann_sample,np.concatenate((ann_sample[1:],[sig_len+1]))):
        if(sym in ann_stoi.keys()):
            mask[sta:end]=ann_stoi[sym]
    return mask

def rescale_segmentation_mask(mask, fs, target_fs):
    factor = target_fs/fs
    timesteps_new = [int(x) for x in np.rint(np.array(range(int(len(mask)*factor)))*fs/target_fs)]
    return mask[timesteps_new]

def create_segment_list(sig_len,ann_sample,ann_annotation,ann_stoi,ann_unk_id,discard_labels=[""]):
    ann_annotation = [x.replace('\x00','') for x in ann_annotation]
    ann_sample = np.array(ann_sample)[np.where(np.isin(ann_annotation,discard_labels)==False)[0]]
    ann_annotation = np.array(ann_annotation)[np.where(np.isin(ann_annotation,discard_labels)==False)[0]]
    
    segment_list=[]
    label_list=[]
    for sym,sta,end in zip(ann_annotation,ann_sample,np.concatenate((ann_sample[1:],[sig_len+1]))):
        segment_list.append((sta,end))
        label_list.append(ann_stoi[sym] if sym in ann_stoi.keys() else ann_unk_id)
    return segment_list, label_list

def rescale_segment_list(segment_list,fs,target_fs):
    factor = target_fs/fs
    return [(int(np.round(a*factor)),int(np.round(b*factor))) for (a,b) in segment_list]

def get_data_annot_stats(data_path,annotation=True,rhythm=True):
    result = []
    for filename in tqdm(list(Path(data_path).glob('**/*.hea'))):
        try:
            _, header = wfdb.rdsamp(str(filename)[:-4])
        except:
            print("Invalid file:",filename)
            continue
        meta = {"filename":filename, "channels":header["sig_name"]}
        if(annotation):
            if(not Path(str(filename)[:-4]+".atr").exists()):
                meta["symbol"]=[]
                if(rhythm):
                    meta["rhythm"]=[]
                else:
                    meta["aux_note"]=[]
            else:
                ann = wfdb.rdann(str(filename)[:-4], 'atr')
                meta["symbol"]= np.unique(ann.symbol)
            
                if(rhythm):
                
                    meta["rhythm"]= np.unique(np.array(ann.aux_note)[np.array(ann.symbol)=="+"])
                else:
                    meta["aux_note"]= np.unique(ann.aux_note)
        result.append(meta)
    df_stats = pd.DataFrame(result)
    
    print('unique channel combinations:')
    unique_channel_comb = [list(x) for x in set(tuple(x) for x in df_stats.channels)]
    for u in unique_channel_comb:
        print(u,np.sum(df_stats.channels.apply(lambda x: x == u) ))
    
    if(annotation):
        unique_symbols, unique_symbols_counts = np.unique([item for sublist in list(df_stats.symbol) for item in sublist],return_counts=True)
        print("\nunique annotations (1 count per file)")
        for us,usc in zip(unique_symbols,unique_symbols_counts):
            print(us,usc)
    if(rhythm and annotation):
        unique_rhythms, unique_rhythms_counts = np.unique([item for sublist in list(df_stats.rhythm) for item in sublist],return_counts=True)
        print("\nunique rhythms (1 count per file)")
        for ur,urc in zip(unique_rhythms,unique_rhythms_counts):
            print(ur,urc)
    return df_stats

############################################
# GENERIC PREPROCESSING METHOD
############################################
def process_physionet_sample_with_annotations(filename,patient_id,channel_labels,ann_stoi, ann_unk_id,discard_labels=[""], rhythm=True, create_segments=True, min_len_segments=100, drop_unk=True, target_fs=100, channels=12, channel_stoi=channel_stoi_canonical,):
    try:
        sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
    except:
        print("Invalid file:",filename)
        return None, None, None
    comments = " ".join(header["comments"])
    
    assert(target_fs<=header['fs'])
    
    #USE df.channels instead of header.sig_names to allow to rename channels before
    data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=channel_labels,fs=header['fs'],target_fs=target_fs,channels=channels)
    #print(header,data[0])
    assert(header['units'][0]=='mV')
    #data = data * 0.001 #fix mV???
    
    ann = wfdb.rdann(str(filename)[:-4], 'atr')
    ann_sample = ann.sample
    #ann_subtype = ann.subtype #for noise etc not used atm
    
    if(rhythm):
        ann_sample = np.array(ann_sample)[np.where(np.array(ann.symbol) == "+")[0]]
        ann_annotation = np.array(ann.aux_note)[np.where(np.array(ann.symbol) == "+")[0]]
    else:
        ann_annotation = ann.symbol

    output_metadata = []
    output_data=[]    
    if(create_segments): #create segments
        segments,segments_label = create_segment_list(len(sigbufs),ann_sample,ann_annotation,ann_stoi,ann_unk_id,discard_labels=discard_labels)
        segments = rescale_segment_list(segments,header["fs"],target_fs)
        if(drop_unk):
            segments = np.array(segments)[np.where(np.array(segments_label)!=ann_unk_id)]
            segments_label = np.array(segments_label)[np.where(np.array(segments_label)!=ann_unk_id)]
        for i,(seg,segl) in enumerate(zip(segments,segments_label)):
            if(seg[1]-seg[0]>0 and (min_len_segments==0 or seg[1]-seg[0]>= min_len_segments)): #discard segments that are too short
                meta={"comments": comments, "data": Path(filename.stem+"_"+str(i)+".npy"), "label":segl, "label_unique": [segl], "start_index":seg[0], "end_index":seg[1], "patient_id":patient_id}
                #np.save(target_root/(filename.stem+"_"+str(i)+".npy"),data[seg[0]:seg[1]])
                output_data.append(data[seg[0]:seg[1]])
                output_metadata.append(meta)
    # just one segment (always do this to be able to stratify based on these)
    mask = create_segmentation_mask(len(sigbufs),ann_sample,ann_annotation,ann_stoi,ann_unk_id,discard_labels=discard_labels)
    mask = rescale_segmentation_mask(mask, header["fs"], target_fs)
    if(drop_unk): # drop unk at start/end of the sequence
        for start_id in range(len(mask)):
            if(mask[start_id]!=ann_unk_id):
                break
        for end_id in reversed(range(len(mask))):
            if(mask[end_id]!=ann_unk_id):
                break
        end_id += 1
        mask = mask[start_id:end_id]
        data = data[start_id:end_id]
        #unk_id somewhere in the middle- ann_stoi probably incorrect
        assert(not ann_unk_id in mask)

    meta={"comments": comments, "data": Path(filename.stem+".npy"), "label":Path(filename.stem+"_ann.npy"), "label_unique":np.unique(mask), "patient_id":patient_id}
    if(create_segments):
        return output_data, output_metadata, meta
        #metadata_single.append(meta)
    else:
        return data, mask, meta
        #np.save(target_root/(filename.stem+".npy"),data)
        #np.save(target_root/(filename.stem+"_ann.npy"),mask)
        #metadata.append(meta)


def prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="mitdb", discard_labels=[""], rhythm=True, create_segments=True, min_len_segments=100, drop_unk=True, target_fs=100, strat_folds=10, channels=12, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    '''
    Two modes:
    * segmentation mask (can be converted into multilabels) or 
    * segments with uniform rhythm 
    
    Note: Possibility to rename channels in df_stats (df_stats.channels will be used) and/or to filter records
    Note: if patient_id is set in df_stats it will be taken into account for stratified splits
    '''
    target_root = Path(".") if target_folder is None else Path(target_folder)
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data is True):
        
        metadata = []
        metadata_single = []
        #for filename in tqdm(list(data_path.glob('**/*.hea'))):
        for sample_id,row in tqdm(df_stats.iterrows(), total=len(df_stats)):
            filename=Path(row["filename"])
            data,mask_or_meta,meta = process_physionet_sample_with_annotations(filename, patient_id=row["patient_id"] if "patient_id" in df_stats.columns else sample_id, channel_labels=row["channels"], ann_stoi=ann_stoi, ann_unk_id=ann_unk_id,discard_labels=discard_labels, rhythm=rhythm, create_segments=create_segments, min_len_segments=min_len_segments, drop_unk=drop_unk, target_fs=target_fs, channels=channels, channel_stoi=channel_stoi)
            if(data is None):
                continue

            if(create_segments):
                for i,(d,m) in zip(data,mask_or_meta):
                    metadata.append(m)
                    np.save(target_root/(filename.stem+"_"+str(i)+".npy"),d)
                metadata_single.append(meta)
            else:
                np.save(target_root/(filename.stem+".npy"),data)
                np.save(target_root/(filename.stem+"_ann.npy"),mask_or_meta)
                metadata.append(meta)
                metadata_single.append(meta)
            
        df =pd.DataFrame(metadata)
        if(create_segments):
            df_single = pd.DataFrame(metadata_single)
            lbl_unique_single = np.unique([item for sublist in list(df_single["label_unique"]) for item in sublist])

        df["dataset"]=dataset_name
        #global label
        lbl_unique = np.unique([item for sublist in list(df["label_unique"]) for item in sublist])
        #stratified_ids = stratify(list(df["label_unique"]), lbl_unique, [1./strat_folds]*strat_folds)
        #df["strat_fold"]=-1
        #idxs = np.array(df.index.values)
        #for i,split in enumerate(stratified_ids):
        #    df.loc[idxs[split],"strat_fold"]=i
        #cv splits SPLIT SEGMENTS CONSISTENT WITH SINGLE VERSION
        df_patients = (df_single if create_segments else df).groupby("patient_id")["label_unique"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: [item for sublist in x for item in sublist]))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify(patients_labels, lbl_unique_single if create_segments else lbl_unique, [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs)
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        df["strat_fold"]=-1
        for i,split in enumerate(stratified_patient_ids):
            df.loc[df.patient_id.isin(split),"strat_fold"]=i
        
        #obtain lbl_itos for consistency reasons
        lbl_itos = [""]*int(1+max(np.unique(list(ann_stoi.values())))) if drop_unk else [""]*int(1+max(np.unique([ann_unk_id]+list(ann_stoi.values()))))
        for k in ann_stoi.keys():
            lbl_itos[ann_stoi[k]]= k if lbl_itos[ann_stoi[k]]=="" else lbl_itos[ann_stoi[k]]+'|'+k
        if(drop_unk is False):
            lbl_itos[ann_unk_id]='UNK' if lbl_itos[ann_unk_id]=="" else lbl_itos[ann_unk_id]+'UNK'
        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std

##################################################################
# SPECIFIC DATASETS
##################################################################

#keep as much as possible for stratification, evaluation etc.
#can map everything apart from 0,1 to bg for training
ann_stoi_af = {"(N":0, "(AFIB":1, "(AFL":2, "(SVTA":3, "(VT":4 }
ann_unk_id_af = 5

###################################################################
# MITDB
###################################################################

#lead selection mitdb c.f. 2003.12009
channel_stoi_mitdb={"mlii":0, "v1":1, "v2":1, "v4":1, "v5":1}

#unique rhythms (1 count per file)
#(AB 1
#(AFIB 10
#(AFL 3
#(B 14
#(BII 2
#(IVR 3
#(N 63
#(NOD 5
#(P 4
#(PREX 2
#(SBR 2
#(SVTA 8
#(T 15
#(VFL 1
#(VT 15
 
def prepare_mitdb(data_path, ann_stoi=ann_stoi_af, ann_unk_id=ann_unk_id_af, create_segments=True, drop_unk=False, target_fs=128, strat_folds=5, channels=2, channel_stoi=channel_stoi_mitdb, target_folder=None, recreate_data=True):

    print("Preparing dataset mitdb.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)
    
    df_stats["ecg_id"]=df_stats.filename.apply(lambda x: x.stem)
    #201 and 202 belong to the same patient
    df_stats["patient_id"]=df_stats.ecg_id.apply(lambda x: "201" if x=="202" else x)

    #discard one patient with strongly deviating lead configuration
    df_stats = df_stats[df_stats.channels.apply(lambda x: x!= ["V5","V2"] )]
    print("\n\nProcessing records...")
    return prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="mitdb", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, strat_folds=strat_folds, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder, recreate_data=recreate_data)
    
##################################################################################
# AFDB
##################################################################################

channel_stoi_afdb={"ecg1":1, "ecg2":0}
#(AFIB 23
#(AFL 8
#(J 3
#(N 21

def prepare_afdb(data_path, ann_stoi=ann_stoi_af, ann_unk_id=ann_unk_id_af, create_segments=False, drop_unk=False, target_fs=128, strat_folds=5, channels=2, channel_stoi=channel_stoi_afdb, target_folder=None, recreate_data=True):

    print("Preparing dataset afdb.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)
    df_stats["ecg_id"]=df_stats.filename.apply(lambda x: x.stem)

    print("\n\nProcessing records...")
    return prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="afdb", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, strat_folds=strat_folds, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder, recreate_data=recreate_data)

#################################################################################
#LTAFDB
#################################################################################

#(AB 46
#(AFIB 83
#(B 21
#(IVR 4
#(N 53
#(SBR 35
#(SVTA 45
#(T 22
#(VT 34
 
def prepare_ltafdb(data_path, ann_stoi=ann_stoi_af, ann_unk_id=ann_unk_id_af, create_segments=False, drop_unk=False, target_fs=128, strat_folds=5, channels=2, channel_stoi=channel_stoi_afdb, target_folder=None, recreate_data=True):

    print("Preparing dataset ltafdb.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)
    df_stats.channels=[['ecg1','ecg2']]*len(df_stats)
    df_stats["ecg_id"]=df_stats.filename.apply(lambda x: x.stem)
    
    print("\n\nProcessing records...")
    return prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="ltafdb", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, strat_folds=strat_folds, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder, recreate_data=recreate_data)


#################################################################################
#VFDB
#################################################################################

#(AFIB 3
#(ASYS 6
#(B 2
#(BI 3
#(HGEA 3
#(N 15
#(NOD 3
#(NOISE 14
#(NSR 1
#(PM 1
#(SBR 1
#(SVTA 3
#(VER 2
#(VF 2
#(VFIB 2
#(VFL 6
#(VT 19
 
def prepare_vfdb(data_path, ann_stoi=ann_stoi_af, ann_unk_id=ann_unk_id_af, create_segments=False, drop_unk=False, target_fs=128, strat_folds=5, channels=2, channel_stoi=channel_stoi_afdb, target_folder=None, recreate_data=True):

    print("Preparing dataset vfdb.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)
    df_stats.channels=[['ecg1','ecg2']]*len(df_stats)
    df_stats["ecg_id"]=df_stats.filename.apply(lambda x: x.stem)
    
    print("\n\nProcessing records...")
    return prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="vfdb", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, strat_folds=strat_folds, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder, recreate_data=recreate_data)

#################################################################################
#CPSC21
#################################################################################

channel_stoi_cpsc21={"i":1, "ii":0}

#(AFIB 682
#(AFL 22
#(N 704

def prepare_cpsc21(data_path, ann_stoi=ann_stoi_af, ann_unk_id=ann_unk_id_af, create_segments=False, drop_unk=False, target_fs=128, strat_folds=5, channels=2, channel_stoi=channel_stoi_cpsc21, target_folder=None, recreate_data=True):

    print("Preparing dataset cpsc21.\nLoading dataset stats...")
    df_stats = get_data_annot_stats(data_path)

    df_stats["ecg_id"]=df_stats.filename.apply(lambda x: x.stem[5:])
    df_stats["patient_id"]=df_stats.ecg_id.apply(lambda x:x.split("_")[0])
    
    print("\n\nProcessing records...")
    return prepare_physionet_dataset_with_annotations(df_stats, ann_stoi, ann_unk_id, dataset_name="cpsc21", discard_labels=[""], rhythm=True, create_segments=create_segments, drop_unk=drop_unk, target_fs=target_fs, strat_folds=strat_folds, channels=channels, channel_stoi=channel_stoi, target_folder=target_folder, recreate_data=recreate_data)


###################################################################################
# icentia
###################################################################################

channel_stoi_icentia={"ecg":0}

ann_stoi_icentia = {"(N":0, "(AFIB":1, "(AFL":2 }
ann_unk_id_icentia = 3

def prepare_icentia(data_path="icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip", ann_stoi=ann_stoi_icentia, ann_unk_id=ann_unk_id_icentia,create_segments=False, drop_unk=False, target_fs=128, strat_folds=10,channel_stoi=channel_stoi_icentia, target_folder=None, recreate_data=True):

    print("Preparing dataset icentia...")

    target_root = Path(".") if target_folder is None else Path(target_folder)
    target_root.mkdir(parents=True, exist_ok=True)

    if(recreate_data):
        with zipfile.ZipFile(data_path, mode="r") as archive:
            lst = archive.namelist()
            lst = [x for x in lst if x.endswith(".hea")]
            df = pd.DataFrame(lst)
            df.columns=["path"]
            df["patient"]=df["path"].apply(lambda x: int(x.split("/")[-1].split("_")[0][1:]))
            df["segment"]=df["path"].apply(lambda x: int(x.split("_")[-1].split(".")[0][1:]))
            metadata=[]
            for pat in tqdm(np.unique(df["patient"])):
                df_selected=df[df.patient==pat].sort_values("segment")
                datas=[]
                masks=[]
                metas=[]
                for l in tqdm(df_selected.path,leave=False):
                    #extract files
                    archive.extract(l, path="tmp_dir/")
                    archive.extract(l[:-3]+"dat", path="tmp_dir/")
                    archive.extract(l[:-3]+"atr", path="tmp_dir/")
                    #process files
                    filename = "tmp_dir/"+l
                    data, mask, meta = process_physionet_sample_with_annotations(filename, patient_id=pat, channel_labels=["ecg"], ann_stoi=ann_stoi, ann_unk_id=ann_unk_id,discard_labels=[""], rhythm=True, create_segments=create_segments, min_len_segments=0, drop_unk=drop_unk, target_fs=target_fs, channels=1, channel_stoi=channel_stoi)
                    datas.append(data)
                    masks.append(mask)
                    metas.append(meta)
                    #remove files
                    os.unlink("tmp_dir/"+l)
                    os.unlink("tmp_dir/"+l[:-3]+"dat")
                    os.unlink("tmp_dir/"+l[:-3]+"atr")
                #save full patient sample
                filename_full = "p"+format(pat, '05d')
                np.save(target_root/(filename_full+".npy"),np.concatenate(datas))
                np.save(target_root/(filename_full+"_ann.npy"),np.concatenate(masks))
                meta=metas[-1]
                meta["data"]=filename_full+".npy"
                metadata.append(meta)
            shutil.rmtree("tmp_dir")
            df=pd.DataFrame(metadata)

        splits_patients = get_stratified_kfolds(np.zeros(len(df)),n_splits=strat_folds,random_state=42)
        df["strat_fold"]=-1
        for i,split in enumerate(splits_patients):
            df.loc[split[-1],"strat_fold"]=i

        #obtain lbl_itos for consistency reasons
        lbl_itos = [""]*int(1+max(np.unique(list(ann_stoi.values())))) if drop_unk else [""]*int(1+max(np.unique([ann_unk_id]+list(ann_stoi.values()))))
        for k in ann_stoi.keys():
            lbl_itos[ann_stoi[k]]= k if lbl_itos[ann_stoi[k]]=="" else lbl_itos[ann_stoi[k]]+'|'+k
        if(drop_unk is False):
            lbl_itos[ann_unk_id]='UNK' if lbl_itos[ann_unk_id]=="" else lbl_itos[ann_unk_id]+'UNK'
        #add means and std
        dataset_add_mean_col(df,data_folder=target_root)
        dataset_add_std_col(df,data_folder=target_root)
        dataset_add_length_col(df,data_folder=target_root)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df, lbl_itos, mean, std, target_root)
    else:
        df, lbl_itos, mean, std = load_dataset(target_root,df_mapped=False)
    return df, lbl_itos, mean, std


##################################
# MIMIC ECG waveforms
#################################

def prepare_mimicecg(data_path="", clip_amp=3, target_fs=100, channels=12, strat_folds=20, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):
    
    if(recreate_data):
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(Path(data_path)/"mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip", 'r') as archive:
            lst = archive.namelist()
            lst = [x for x in lst if x.endswith(".hea")]

            meta = []
            for l in tqdm(lst):
                archive.extract(l, path="tmp_dir/")
                archive.extract(l[:-3]+"dat", path="tmp_dir/")
                filename = Path("tmp_dir")/l
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            
                tmp={}
                tmp["data"]=filename.parent.parent.stem+"_"+filename.parent.stem+".npy" #patientid_study.npy
                tmp["study_id"]=int(filename.stem)
                tmp["subject_id"]=int(filename.parent.parent.stem[1:])
                tmp['ecg_time']= datetime.datetime.combine(header["base_date"],header["base_time"])
                tmp["nans"]= list(np.sum(np.isnan(sigbufs),axis=0))#save nans channel-dependent
                if(np.sum(tmp["nans"])>0):#fix nans
                    fix_nans_and_clip(sigbufs,clip_amp=clip_amp)
                elif(clip_amp>0):
                    sigbufs = np.clip(sigbufs,a_max=clip_amp,a_min=-clip_amp)

                data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
                
                assert(target_fs<=header['fs'])
                np.save(target_folder/tmp["data"],data)
                meta.append(tmp)
                
                os.unlink("tmp_dir/"+l)
                os.unlink("tmp_dir/"+l[:-3]+"dat")
                shutil.rmtree("tmp_dir")

        df = pd.DataFrame(meta)

        #random split by patients
        #unique_patients = np.unique(df.subject_id)
        #splits_patients = get_stratified_kfolds(np.zeros(len(unique_patients)),n_splits=strat_folds,random_state=42)
        #df["fold"]=-1
        #for i,split in enumerate(splits_patients):
        #    df.loc[df.subject_id.isin(unique_patients[split[-1]]),"fold"]=i
        
        #add means and std
        dataset_add_mean_col(df,data_folder=target_folder)
        dataset_add_std_col(df,data_folder=target_folder)
        dataset_add_length_col(df,data_folder=target_folder)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        lbl_itos=[]
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std

##################################
# PTB-Diagnostic again
#################################

def prepare_ptbv2(data_path="", min_cnt=9, target_fs=100, channels=12, strat_folds=5, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True):

    if(recreate_data):
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        def extract_comments(file_path):
            # Read comments from HEA file using wfdb
            record = wfdb.rdrecord(file_path[:-4], physical=False)
            return record.comments

        df=[]
        # Loop through subfolders and extract comments
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith('.hea'):
                    file_path = os.path.join(root, file)
                    comments = extract_comments(file_path)

                    # Append data to the DataFrame
                    df += [{'filename': file_path, 'Comments': comments}]

        df=pd.DataFrame(df)

        for _,row in df.iterrows():
            sigbufs, header = wfdb.rdsamp(str(row["filename"])[:-4])
            data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
            
            stem=Path(row["filename"]).stem
            assert(target_fs<=header['fs'])
            np.save(target_folder/stem,data)
        df["data"]=df["filename"].apply(lambda x: Path(x).stem+".npy")
        df["patient_id"]=df["filename"].apply(lambda x: int(Path(x).parent.stem.replace("patient","")))
        
        #extract diagnoses from comments
        df["Comments"]=df["Comments"].apply(lambda x: [l.split(":") for l in x])
        df["Comments"]=df["Comments"].apply(lambda x: {y[0]:y[1] for y in x})


        for k in df["Comments"].iloc[0].keys():
            df[k]=df["Comments"].apply(lambda x: x[k])


        df = df.drop(columns=['Comments'])
        df = df.drop(columns=['Diagnose'])

        df["Additional diagnoses"]=df["Additional diagnoses"].apply(lambda x: x.split(","))

        df["Combined diagnoses"]=df.apply(lambda row:[row["Reason for admission"]]+row["Additional diagnoses"],axis=1)

        def clean_diags(lst):
            lst=[x.strip() for x in lst]
            blacklist=["no","n/a","unknown", "M. Bechterew", "M. Basedow"]
            lst=[x for x in lst if not (x in blacklist)]
            return lst

        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: clean_diags(x))

        df['ECG date']=df['ECG date'].apply(lambda x: np.nan if x.strip()=="n/a" else x.strip())
        df['ECG date'] = pd.to_datetime(df['ECG date'], format='%d/%m/%Y')

        df['Infarction date (acute)']=df['Infarction date (acute)'].apply(lambda x: np.nan if x.strip()=="n/a" else x.strip())
        df['Infarction date (acute)'] = pd.to_datetime(df['Infarction date (acute)'], format='%d-%b-%y')

        df['Catheterization date']=df['Catheterization date'].apply(lambda x: np.nan if (x.strip()=="n/a" or x.strip()=="-") else x.strip())
        df['Catheterization date'] = pd.to_datetime(df['Catheterization date'], format='%d-%b-%y')

        df['days_since_infarction'] = (df['ECG date'] - df['Infarction date (acute)']).dt.days
        df['days_since_catheterization'] = (df['ECG date'] - df['Catheterization date']).dt.days

        #https://emedicine.medscape.com/article/1960472-overview?form=fpf
        df['acute_MI']= df['days_since_infarction'].apply(lambda x: x>=0 and x<=3)
        df["catheterized"]=df.apply(lambda x: x["days_since_infarction"]>=0 and x["days_since_catheterization"]<=x["days_since_infarction"],axis=1)

        df["MI"]=df["Combined diagnoses"].apply(lambda x: "Myocardial infarction" in x)

        cleanup_map={
            "Hyperlipoproteinemia Type IIa":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Type IV":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Type IIb":"Hyperlipoproteinemia",
            "Hyperlipemia":"Hyperlipoproteinemia",
            "Hyperlipoproteinemia Typ IIa":"Hyperlipoproteinemia",
            "Postop. Thyriodectomy":"Postop. Thyroidectomy",
            "Coronary artery disease (PTCA+Stent) ":"Coronary artery disease",
            "Hyperlipoproteinemia Type IV b":"Hyperlipoproteinemia",
            "peripheral atherosclerosis":"Peripheral atherosclerosis",
            "Recurrent sustained ventricular tachycardia":"Ventricular tachycardia",
            "Ventricular tachycardias":"Ventricular tachycardia",
            "Recurrent ventricular tachycardias":"Ventricular tachycardia",
            "Recurrent ventricular tachycardia":"Ventricular tachycardia",
            "Hyperlipidemia":"Hyperlipoproteinemia",
            "Mitral regurgitation (grade 2)":"Mitral regurgitation",
            "Hypertrophic obstructive cardiomyopathy (intraventricular gradient 100-160mmHg)":"Hypertrophic obstructive cardiomyopathy",
            "Hypertrophic obstructive cardiomyopathy (intraventricular gradient 70-90mmHg)":"Hypertrophic obstructive cardiomyopathy"  
        }

        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: [y.strip() for y in x])
        df["Combined diagnoses"]=df["Combined diagnoses"].apply(lambda x: [(cleanup_map[y] if y in cleanup_map.keys() else y) for y in x])

        def mistatus(mi,acute,catheterized):
            if(not mi):
                return ""
            res=""
            if(not acute):
                res+=" old"
            else:
                res+=" acute"
                if(catheterized):
                    res+=" catheterized"
            return res

        df["MI_status"]=df.apply(lambda row:mistatus(row["MI"],row["acute_MI"],row["catheterized"]), axis=1)
        df["Combined diagnoses"]=df.apply(lambda row: [(y+row["MI_status"] if y== "Myocardial infarction" else y) for y in row["Combined diagnoses"]] ,axis=1)

        diag,cnt = np.unique([x for xs in list(df["Combined diagnoses"]) for x in xs], return_counts=True)
        idx=np.argsort(cnt)[::-1]

        label_itos=[]
        res=[]
        for d,c in zip(diag[idx],cnt[idx]):
            if(c>=min_cnt):
                label_itos.append(d)
            res.append({"diag":d,"count":c})
            print(d,c)
        #df_stats=pd.DataFrame(res)
        #df_stats.to_excel("ptb_stats.xlsx")

        label_stoi={s:i for i,s in enumerate(label_itos)}
        df["label"]=df["Combined diagnoses"].apply(lambda x: [label_stoi[y] for y in x if y in label_itos])

        #random split by patients
        unique_patients = np.unique(df.patient_id)
        splits_patients = get_stratified_kfolds(np.zeros(len(unique_patients)),n_splits=strat_folds,random_state=42)
        df["strat_fold"]=-1
        for i,split in enumerate(splits_patients):
            df.loc[df.patient_id.isin(unique_patients[split[-1]]),"strat_fold"]=i
        
        #add means and std
        dataset_add_mean_col(df,data_folder=target_folder)
        dataset_add_std_col(df,data_folder=target_folder)
        dataset_add_length_col(df,data_folder=target_folder)
        #dataset_add_median_col(df_ptb_xl,data_folder=target_root_ptb_xl)
        #dataset_add_iqr_col(df_ptb_xl,data_folder=target_root_ptb_xl)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std
        

##################################
# HEEDB waveforms
#################################
def process_ecg_file_heedb(row, data_path, target_folder, target_fs, channels, channel_stoi, clip_amp, recreate_numpy):
    try:
        filename = Path(data_path) / "ECG/WFDB" / row["FileName"]
        filename_new = filename.stem + ".npy"
        
        # Check if we can skip processing
        if not recreate_numpy and (Path(target_folder) / filename_new).exists():
            return {"file_source": row["file_source"], "data": str(filename_new)}
        
        # Validate header file exists
        if not filename.exists():
            print(f"Header {filename} does not exist.")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Validate signal file exists
        signal_file = filename.parent / (filename.stem + ".mat")
        if not signal_file.exists():
            print(f"Signal file corresponding to {filename} does not exist.")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Read the signal
        try:
            sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
        except Exception as e:
            error_message = str(e) if str(e) else "Unknown error reading signal file"
            print(f"Warning: could not read {filename}. Error: {error_message}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Check if header contains required fields
        if 'fs' not in header or 'sig_name' not in header:
            print(f"Missing required fields in header for {filename}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Handle NaNs and clipping
        if np.sum(np.isnan(sigbufs)) > 0:
            try:
                fix_nans_and_clip(sigbufs, clip_amp=clip_amp)
            except Exception as e:
                print(f"Error fixing NaNs in {filename}: {str(e)}")
                return {"file_source": row["file_source"], "data": "nan"}
        elif clip_amp > 0:
            sigbufs = np.clip(sigbufs, a_max=clip_amp, a_min=-clip_amp)
        
        # Validate target frequency
        if target_fs > header['fs']:
            print(f"Target frequency {target_fs} is higher than original {header['fs']} for {filename}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Resample data
        try:
            data = resample_data(
                sigbufs=sigbufs,
                channel_stoi=channel_stoi,
                channel_labels=header['sig_name'],
                fs=header['fs'],
                target_fs=target_fs,
                channels=channels
            )
        except Exception as e:
            print(f"Error resampling data for {filename}: {str(e)}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        # Save processed data
        try:
            output_path = Path(target_folder) / filename_new
            np.save(output_path, data)
        except Exception as e:
            print(f"Error saving data for {filename}: {str(e)}")
            return {"file_source": row["file_source"], "data": "nan"}
        
        return {"file_source": row["file_source"], "data": str(filename_new)}
        
    except Exception as e:
        # Make sure we get a string representation of the error
        error_message = str(e) if str(e) else "Empty exception message"
        traceback_info = traceback.format_exc()
        print(f"Error processing {row.get('FileName', 'unknown file')}: {error_message}")
        print(f"Traceback: {traceback_info}")
        return {"file_source": row.get("file_source", "unknown"), "data": "nan"}
    
#['S0001-1987', 'S0001-1988', 'S0001-1994', 'S0001-2006', 'S0001-2007', 'S0001-2011', 'S0001-2013', 'S0001-2015', 'S0001-2016', 'S0001-2017', 'S0001-2018', 'S0001-2019', 'S0001-2020', 'S0001-2021', 'S0002-1993', 'S0002-1997', 'S0002-1998', 'S0002-2000', 'S0002-2003', 'S0002-2004', 'S0002-2008', 'S0002-2009', 'S0002-2013', 'S0002-2015', 'S0002-2016', 'S0002-2018', 'S0002-2020', 'S0002-2022', 'S0003-1990', 'S0003-1994', 'S0003-2009', 'S0003-2010', 'S0003-2013', 'S0004-2019']
def prepare_heedb(data_path="", partition="S0002-1998",clip_amp=3, target_fs=100, channels=12, strat_folds=20, channel_stoi=channel_stoi_canonical, target_folder=None, recreate_data=True, max_workers=8, recreate_numpy=False, skip_metadata=False):

    def prepare_metadata(partition,data_path):
        print("Reading metadata table...")
        df_demo=pd.read_csv(Path(data_path)/"Metadata/demographics_ECG.csv", low_memory=False)
        df_demo["partition"]=df_demo.FileName.apply(lambda x: "-".join(x.split("/")[1:3]))
        df_demo=df_demo[df_demo.partition==partition]
        df_demo["file_source"]=df_demo.FileName.apply(lambda x:x.split("/")[-1][:-len(".hea")])
        print(len(df_demo),"samples in the selection.")
        print("Reading ECG interpretation statements...")
        df_diag= pd.read_csv(Path(data_path)/"Metadata/ECG_Interpretations.psv",delimiter="|")
        df_diag["diagnostic_codes"]=df_diag["diagnostic_codes"].apply(lambda x: [int(y) for y in x.split(",")])
        labels= [x for xs in list(df_diag["diagnostic_codes"]) for x in xs]
        lbls, cnts = np.unique(labels, return_counts=True)
        lbl_itos = lbls[np.argsort(cnts)[::-1]]
        print("Reading duplicated list...")
        df_dupl=pd.read_csv(Path(data_path)/"Metadata/duplicated.csv")
        dupl_lst = list(df_dupl.file_name.apply(lambda x:x.split("/")[-1]))
        print("Joining tables...")
        df_demo=df_demo.join(df_diag.set_index("file_source"),on="file_source",how="left")
        #remove nans (also due to non-matching entries)
        df_demo["diagnostic_codes"]=df_demo["diagnostic_codes"].apply(lambda x: x if isinstance(x,list) else [])
        df_demo["PatientRace"]=df_demo["PatientRace"].replace({np.nan: 'Unavailable'})
        df_demo["SexDSC"]=df_demo["SexDSC"].replace({np.nan: 'Unknown'})
        df_demo=df_demo[~df_demo.file_source.isin(dupl_lst)].copy()#remove duplicated
        del df_diag
        return df_demo, lbl_itos
    
    
    if(recreate_data):
        print("Processing partition",partition)
        data_path = Path(data_path)
        target_folder = Path(target_folder)/partition
        (target_folder).mkdir(parents=True, exist_ok=True)
        if(not skip_metadata):
            print("Preparing metadata...")
            df, lbl_itos = prepare_metadata(partition,data_path)
            print("\nDone. Preparing signals... ")
        else:
            print("Preparing file list...")
            extra_path = "ECG/WFDB/"+"/".join(partition.split("-"))
            hea_files = []
            for root, dirs, filenames in os.walk(str(data_path/extra_path)):
                for filename in filenames:
                    if filename.endswith('.hea'):
                        hea_files.append(os.path.join(root, filename))
            #hea_files = list((data_path/extra_path).rglob('*.hea'))
            df = pd.DataFrame(hea_files, columns=['FileName'])
            df['FileName'] = df['FileName'].astype(str)
            df["file_source"]=df.FileName.apply(lambda x:x.split("/")[-1][:-len(".hea")])
            
        filenames_new = []

        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing jobs
            futures = {
                executor.submit(
                    process_ecg_file_heedb, 
                    row, 
                    data_path, 
                    target_folder, 
                    target_fs, 
                    channels, 
                    channel_stoi, 
                    clip_amp,
                    recreate_numpy
                ): row for _, row in df.iterrows()
            }
            
            # Tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    filenames_new.append(result)
        
        df_files = pd.DataFrame(filenames_new)
        
        if(skip_metadata):
            return df_files, [], None, None
        
        df=df.join(df_files.set_index("file_source"),on="file_source",how="left")
        
        if(len(df[df["data"]=="nan"])>0):
            print("The following files could not be processed:",list(df[df["data"]=="nan"]["FileName"]),"\nThe files will be skipped.")
            df=df[df["data"]!="nan"].copy()

        #fold asignments
        print("Preparing stratified fold distribution...")
        lbl_itos_strat = [str(x) for x in lbl_itos]+["sex_"+x for x in df["SexDSC"].unique()]+["race_"+x for x in df["PatientRace"].unique()]+["age_unknown","age_18","age_30","age_40","age_50","age_60","age_70","age_80","age_90","age_100"]
        def age_to_str(age):
            if(np.isnan(age) or age<0):
                return "age_unknown"
            elif(age<18):
                return "age_18"
            elif(age<30):
                return "age_30"
            elif(age<40):
                return "age_40"
            elif(age<50):
                return "age_50"
            elif(age<60):
                return "age_60"
            elif(age<70):
                return "age_70"
            elif(age<80):
                return "age_80"
            elif(age<90):
                return "age_90"
            else:
                return "age_100"
           
        df["label_strat"] = df.apply(lambda row: [str(y) for y in row["diagnostic_codes"]] + 
                ["sex_" + row["SexDSC"]] + 
                ["race_" + row["PatientRace"]] + 
                [age_to_str(row["Age"])],
                axis=1)
        df_patients = df.groupby("BDSPPatientID")["label_strat"].apply(lambda x: list(x))
        patients_ids = list(df_patients.index)
        patients_labels = list(df_patients.apply(lambda x: np.concatenate(x)))
        patients_num_ecgs = list(df_patients.apply(len))

        stratified_ids = stratify_batched(patients_labels, lbl_itos_strat, [1./strat_folds]*strat_folds, samples_per_group=patients_num_ecgs,verbose=True,batch_size=min(10000,len(df_patients)))#todo fix this in stratify
        stratified_patient_ids = [[patients_ids[i] for i in fold] for fold in stratified_ids]

        print("Applying fold assignments...")
        df["strat_fold"]=-1 #unlabeled will end up in fold -1
        for i,split in enumerate(stratified_patient_ids):
            df.loc[df.BDSPPatientID.isin(split),"strat_fold"]=i
        print("Fold distribution:",df.strat_fold.value_counts())
        print("In total",len(df),"records from ",len(df.BDSPPatientID.unique()),"patients.")
        df.drop("label_strat",axis=1, inplace=True)

        #add means and std
        #print("Calculating dataset stats...")
        #dataset_add_mean_col(df,data_folder=target_folder)
        #dataset_add_std_col(df,data_folder=target_folder)
        #dataset_add_length_col(df,data_folder=target_folder)

        #save means and stds
        #mean, std = dataset_get_stats(df)
        #dummies for now
        mean = np.zeros((12),dtype=np.float32)
        std = np.ones((12),dtype=np.float32)
        

        #save
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std
