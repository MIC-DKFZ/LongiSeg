# How to use LongiSeg
LongiSeg inherits nnU-Net’s self-configuring capabilities, allowing it to easily adapt to new datasets. It extends nnU-Net to handle medical image time series, leveraging temporal relationships to improve segmentation accuracy.

## Dataset Format
LongiSeg expects the same structured [dataset format](dataset_format.md) as nnU-Net. Datasets can either be saved in the same
folders as nnU-Net or in LongiSeg's own folder (`LongiSeg_raw`, `LongiSeg_preprocessed` and `LongiSeg_results`).
In contrast to nnU-Net, LongiSeg expects an additional `patientsTr.json` file in the dataset folder. This file lists patient IDs and their corresponding scans in chronological order.

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── patientsTr.json
    ├── imagesTr
    ├── imagesTs  # optional
    └── labelsTr

This json file should have the following structure:

    {
        "patient_1": [
            "patient_1_scan_1",
            "patient_1_scan_2",
            ...
        ],
        "patient_2": [
            "patient_2_scan_1",
            "patient_2_scan_2",
            ...
        ],
        ...
    }

## Experiment planning and preprocessing
To run experiment planning and preprocessing of a new dataset, simply run
```bash
LongiSeg_plan_and_preprocess -d DATASET_ID
```
or if you prefer to keep things separate, you can also use `LongiSeg_extract_fingerprint`, `LongiSeg_plan_experiment` 
and `LongiSeg_preprocess` (in that order). We refer to the [nnU-Net documentation](how_to_use_nnunet.md#experiment-planning-and-preprocessing) for additional details on experiment planning and preprocessing.

## Training
To train a model using LongiSeg, simply run
```bash
LongiSeg_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD
```

By default, LongiSeg uses the `LongiSegTrainer`, which integrates the temporal dimension by concatenating multi-timepoint images as additional input channels. Other trainers are available with the -tr option:

- `nnUNetTrainerLongi`: A modified `nnUNetTrainer` where training data is split **at the patient level** instead of per scan.
- `LongiSegTrainerDiffWeighting`: A longitudinal trainer that incorporates the **Difference Weighting Block** for temporal feature fusion.

Other options for training are available as well (`LongiSeg_train -h`).

## Inference
Inference with LongiSeg works in a similar way to the [nnU-Net inference](how_to_use_nnunet.md#run-inference), with the added requirement of specifying a patient file (-pat) in either the `LongiSeg_predict` or `LongiSeg_predict_from_modelfolder` commands. The patient file needs to detail the patient structure in the same way as during training. Only cases present in both the input folder **and** `patients.json` will be processed during inference!

## Evaluation
By default LongiSeg performs nnU-Net's standard evaluation on the 5-fold cross validation. This, however, does not account for individual patients and only calculates a handful of metrics. LongiSeg extends nnU-Net’s evaluation by incorporating additional metrics that provide a more comprehensive assessment of segmentation performance, including volumetric, surface-based, distance-based, and detection metrics.
To run evaluation with LongiSeg, use
```bash
LongiSeg_evaluate_folder GT_FOLDER PRED_FOLDER -djfile /path/to/dataset.json -pfile /path/to/plans.json -patfile /path/to/patients.json
```

If no patient file is provided, LongiSeg will default to standard nnU-Net evaluation while incorporating the additional metrics.