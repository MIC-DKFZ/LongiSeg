# Verified lesion tracking with LongiSeg

This page describes the lesion tracking workflow introduced with LongiSeg 2.0. Instead of segmenting a whole scan,
the model segments **one lesion at a time** in a follow-up scan, using a prompt point in the follow-up scan together
with the baseline appearance of that lesion as longitudinal context.

Tracking a lesion from a baseline to a follow-up scan works in three steps:

1. The lesion is marked in the baseline scan by a point, `bl_point`.
2. A registration based method propagates that point into the follow-up scan, giving `fu_point_prop`.
3. A clinician verifies the propagated point and corrects it where it is wrong, giving `fu_point`.

Segmenting with `fu_point_prop` is *automatic tracking*, segmenting with `fu_point` is *verified tracking*.

For general LongiSeg usage (installation, paths, standard longitudinal segmentation) see
[how_to_use_longiseg.md](how_to_use_longiseg.md).

## Dataset format

Tracking datasets follow the [LongiSeg dataset format](how_to_use_longiseg.md#dataset-format) and add a
`trackingTr.json` file:

    Dataset123_PanTrack/
    ├── dataset.json
    ├── patientsTr.json
    ├── trackingTr.json
    ├── imagesTr
    └── labelsTr

One convention differs from standard LongiSeg datasets: **every lesion carries its own label id** in `labelsTr`
(1, 2, 3, ...) instead of all lesions sharing one foreground class. Corresponding lesions keep the same id across
timepoints, which is how they are matched. Scan names are free, they are only referenced through `trackingTr.json`.

`trackingTr.json` maps every patient to the lesions that are tracked from a baseline to a follow-up scan:

    {
        "patient_1": {
            "1": {
                "img_bl": "patient_1_scan_0",
                "img_fu": "patient_1_scan_1",
                "bl_point": [x, y, z],
                "fu_point_prop": [x, y, z],
                "fu_point": [x, y, z],
                "merged_lesions": [1]
            },
            "2": { ... }
        },
        ...
    }

- The keys of the inner dict are the **label ids of the lesions in the baseline scan**.
- `merged_lesions` lists the label ids in the **follow-up** scan that correspond to this baseline lesion. It holds
  more than one id when several lesions merged into one, and `[0]` when the lesion disappeared in the follow-up
  scan. Lesions that merged are evaluated together.
- All three points are voxel coordinates **in the original image space**, in `(x, y, z)` order (i.e. reversed with
  respect to the numpy axis order). They are resampled to the target spacing during preprocessing. A point that does
  not exist is `NaN`.

To track a lesion through more than two timepoints, store a **list** of such mappings instead, one entry per
consecutive baseline/follow-up pair:

    {
        "patient_2": [
            { "1": { "img_bl": "patient_2_scan_0", "img_fu": "patient_2_scan_1", ... } },
            { "1": { "img_bl": "patient_2_scan_1", "img_fu": "patient_2_scan_2", ... } }
        ]
    }

### The three points

**`bl_point`** marks the lesion in the **baseline** scan, either as the centroid of the annotated baseline lesion or
as the click a clinician placed on it. It is what identifies the lesion to be tracked, and it is used twice: it is
turned into the baseline gaussian prompt channel, and it positions the baseline patch, which is cropped such that
`bl_point` ends up at the same position inside the patch as the follow-up prompt does in the follow-up patch, so
both patches are aligned on the lesion.

**`fu_point_prop`** is `bl_point` propagated into the **follow-up** scan by the registration based method. It is the
prompt of *automatic tracking* (`--mode automatic`) and is used as is, without a clinician ever looking at it. It
also decides whether a lesion takes part at all: a lesion with no propagated point is skipped by
`LongiSeg_predict_tracking`, by the validation at the end of a training and by the evaluation.

**`fu_point`** marks the lesion in the **follow-up** scan, either as the centroid of the annotated follow-up lesion
or as the propagated point after a clinician corrected it. It is the prompt of *verified tracking* (`--mode manual`).

During **training** the prompts are sampled rather than taken verbatim, so that the model sees more than one point
per lesion and does not rely on the prompt sitting exactly at a centroid. The follow-up prompt alternates between
`fu_point_prop` and a random voxel of the lesion, weighted towards its center, and the baseline prompt is a random
voxel of the baseline lesion, weighted the same way. `fu_point` is used whenever no propagated point exists, so it
matters for training as well, not just for verified inference.

## Experiment planning and preprocessing

Tracking needs its own fingerprint extractor and preprocessor. In contrast to the standard pipeline, images are
**not** cropped to their nonzero region, so that the points of baseline and follow-up scan stay in a common
coordinate system:

```bash
LongiSeg_plan_and_preprocess -d DATASET_ID \
    -fpe DatasetFingerprintExtractorLongiSegTrack \
    -preprocessor_name LongiSegTrackingPreprocessor
```

The preprocessor writes one `{patient}.json` with the resampled points next to the preprocessed data and into
`gt_segmentations`, where the validation at the end of a training picks it up.

For **synthetic longitudinal pretraining** datasets there are no tracking annotations; corresponding lesions instead
share the same label id in both scans of a patient, and both prompts are sampled from the lesion directly. Use the
pretraining preprocessor, which also drops patients without a lesion present in all of their scans:

```bash
LongiSeg_plan_and_preprocess -d DATASET_ID \
    -fpe DatasetFingerprintExtractorLongiSegTrack \
    -preprocessor_name LongiSegTrackingPretrainingPreprocessor
```

## Training

```bash
LongiSeg_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr LongiSegTrainerTracking
```

The following tracking trainers are available:

| Trainer | Purpose |
|---|---|
| `LongiSegTrainerTracking` | early prompt fusion (`LongiUNetTracking`) |
| `LongiSegTrainerTrackingDiffWeighting` | additionally uses latent temporal difference weighting (`LongiUNetTrackingDiffWeighting`) |
| `LongiSegTrainerTrackingPretrain`, `LongiSegTrainerTrackingDiffWeightingPretrain` | large-scale synthetic longitudinal pretraining. Uses a single 95:5 split instead of a 5-fold cross-validation and skips the final validation |
| `LongiSegTrainerTrackingFinetuning`, `LongiSegTrainerTrackingDiffWeightingFinetuning` | finetuning of a pretrained checkpoint, with a lowered initial learning rate of 1e-3 |

To finetune a pretrained model, pass the checkpoint with `-pretrained_weights`:

```bash
LongiSeg_train DATASET_NAME_OR_ID 3d_fullres FOLD \
    -tr LongiSegTrainerTrackingDiffWeightingFinetuning \
    -pretrained_weights /path/to/pretrained/checkpoint_final.pth
```

The validation at the end of a training runs automatic tracking, i.e. it prompts with `fu_point_prop`.

## Inference

```bash
LongiSeg_predict_tracking \
    --images_path /path/to/imagesTs \
    --labels_path /path/to/labelsTs \
    --output_path /path/to/output \
    --model_path /path/to/trained_model_folder \
    --tracking_path /path/to/trackingTs.json \
    --dataset_json_path /path/to/dataset.json \
    --mode automatic
```

`--mode automatic` prompts with the propagated point (`fu_point_prop`), `--mode manual` with the verified point
(`fu_point`). `--labels_path` provides the baseline segmentation that gives the model the lesion appearance at
baseline; without it the model only receives the prompt points. One file per tracked lesion is written, named
`{follow-up scan}_lesion_{baseline lesion id}{file_ending}`.

## Evaluation

```bash
LongiSeg_evaluate_tracking GT_FOLDER PRED_FOLDER \
    -djfile /path/to/dataset.json \
    -pfile /path/to/plans.json \
    -tfile /path/to/trackingTs.json
```

`-tfile` is the tracking json of the dataset, i.e. the same file that was passed to `LongiSeg_predict_tracking`. It
tells the evaluation which lesions are tracked into which follow-up scan.

The validation at the end of a training does not need it: there the tracking info is taken from the per patient json
files that `LongiSegTrackingPreprocessor` writes next to the reference segmentations in
`LongiSeg_preprocessed/DATASET/gt_segmentations`.

Metrics are reported per tracked lesion (Dice, Recall, Precision, and false negative / false positive volume), and
lesions that merged in the follow-up scan are evaluated together.
