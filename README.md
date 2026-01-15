# [A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data and Clinician Benchmarking](https://arxiv.org/abs/2601.06645)

[![arXiv](https://img.shields.io/badge/arXiv-2407.17856-b31b1b.svg)](https://arxiv.org/abs/2601.06645)



## Clinical Setting

![alt text](https://github.com/AI4HealthUOL/MDS-ICU/blob/main/reports/Fig1.png?style=centerme)

In this study, conducted within the context of the intensive care unit, we introduce a state-of-the-art biomedical multimodal benchmark for patient deterioration prediction, which we named Medical Decission Support within the ICU contest (**MDS-ICU**).

The datasets include various patient predictors collected at irregular window intervals during the patient ICU stay, such as:
- Demographics
- Biometrics
- Vital parameter trends
- Laboratory value trends
- Surgeries
- Supporting devices
- ECG waveforms

Similarly, the dataset investigate four different target groups that encompasses 33 labels for which we investigate the discriminative sigfinicance that the inclusion of raw wavefroms (ECG) adds to the prediction model:
- Mortality
- Medications
- Clinical deterioration
- Organ dysfunction
  

## Clinician benchmark

![alt text](https://github.com/AI4HealthUOL/MDS-ICU/blob/main/reports/clinician_benchmark.png?style=centerme)

Importantly, we conducted a clinician benchmark in which we evaluate the prediction of 4 clinicians with relevant ICU experiene as well as 2 LLMs against our model. We concluded that in the first benchmark (model vs clinician/LLMs) the model outperform clinicians and LLMs, however, in the second benchmark (model + clinician/LLMs) where we provide the model probabilities to the clinicians and LLMs along the feature set, they will in the majority of the cases improve their performance even overcoming the same model, concluding that our model best serves as a decision support system rather than a standalone decision tool. 


## Reference
```bibtex
@misc{alcaraz2026multimodaldeeplearningframework,
      title={A Multimodal Deep Learning Framework for Predicting ICU Deterioration: Integrating ECG Waveforms with Clinical Data and Clinician Benchmarking}, 
      author={Juan Miguel López Alcaraz and Xicoténcatl López Moran and Erick Dávila Zaragoza and Claas Händel and Richard Koebe and Wilhelm Haverkamp and Nils Strodthoff},
      year={2026},
      eprint={2601.06645},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2601.06645}, 
}
