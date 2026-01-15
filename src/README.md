## Data 

You can preprocess the MDS-ICU dataset on your own, first, download from the following databases the following files and place them under data/, then run full_preprocessing.py

- From MIMIC-IV-ECG-ICD: records_w_diag_icd10.csv
- From MIMIC-IV-ED: vitalsign.csv.gz
- From MIMIC-IV: icustays.csv.gz, admissions.csv.gz, d_items.csv.gz, chartevents.csv.gz, d_labitems.csv.gz, labevents.csv.gz, omr.csv.gz, inputevents.csv.gz, procedureevents.csv.gz, outputevents.csv.gz, services.csv.gz 
- From MIMIC-IV-ECG: mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip


## Experiments

We provide a convenient pipeline for our multimodal (ECG waveforms + tabular) experiments replication. The next commands would train and test model:

```
python main_all.py --config-name config_supervised_multimodal_mdsicu.yaml
```

## Output and Results
The results, including model performance metrics, as well as validation and test predictions will be saved automatically in the the current directory.

For further inquiries, please open an issue.
