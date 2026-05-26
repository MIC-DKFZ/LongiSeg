import argparse
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch._dynamo import OptimizedModule
import torch.nn.functional as F

from batchgenerators.utilities.file_and_folder_operations import load_json

from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor


def generated_sparse_to_dense_point_rescaled_gauss(point: list[int], shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    gauss_blob = np.zeros(shape, dtype=np.float32)
    gauss_blob[tuple(point)] = 1.0
    gauss_blob = gaussian_filter(gauss_blob, sigma=sigma)
    gauss_blob /= gauss_blob[tuple(point)]
    return gauss_blob


def predict_patch(bl_data: torch.Tensor, bl_seg: Optional[torch.Tensor], bl_point: list[int], bl_lesion: int,
                  fu_data: torch.Tensor, fu_point: list[int], predictor: LongiSegPredictor, patch_size: list[int]):

    bl_shape = bl_data.shape[1:]
    fu_shape = fu_data.shape[1:]

    bl_bbox_lbs = []
    bl_bbox_ubs = []
    fu_bbox_lbs = []
    fu_bbox_ubs = []

    for i in range(3):
        if fu_point[i] >= patch_size[i] // 2 and fu_point[i] < fu_shape[i] - patch_size[i] // 2:
            fu_lbs = fu_point[i] - patch_size[i] // 2
        elif fu_point[i] < patch_size[i] // 2 and patch_size[i] <= fu_shape[i]:
            fu_lbs = 0
        elif fu_point[i] >= fu_shape[i] - patch_size[i] // 2 and patch_size[i] <= fu_shape[i]:
            fu_lbs = fu_shape[i] - patch_size[i]
        elif patch_size[i] > fu_shape[i]:
            fu_lbs = -(patch_size[i] - fu_shape[i]) // 2
        else:
            raise RuntimeError(f"Unexpected Combination of fu_point {fu_point}, "
                                f"patch_size {patch_size}, data_shape {fu_shape}")
        bl_lbs = fu_lbs + (bl_point[i] - fu_point[i])
        fu_bbox_lbs.append(fu_lbs)
        fu_bbox_ubs.append(fu_lbs + patch_size[i])
        bl_bbox_lbs.append(bl_lbs)
        bl_bbox_ubs.append(bl_lbs + patch_size[i])

    valid_fu_bbox_lbs = np.clip(fu_bbox_lbs, a_min=0, a_max=None)
    valid_fu_bbox_ubs = np.minimum(fu_shape, fu_bbox_ubs)
    valid_bl_bbox_lbs = np.clip(bl_bbox_lbs, a_min=0, a_max=None)
    valid_bl_bbox_ubs = np.minimum(bl_shape, bl_bbox_ubs)

    fu_slice_data = tuple([slice(0, fu_data.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
    fu_data_patch = fu_data[fu_slice_data]

    bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
    bl_data_patch = bl_data[bl_slice_data]

    if bl_seg is not None:
        bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
        bl_seg_patch = bl_seg[bl_slice_seg]
        bl_seg_patch = torch.where(bl_seg_patch == int(bl_lesion), 1, 0).to(bl_data_patch.dtype)
    else:
        bl_slice_seg = None
        bl_seg_patch = torch.zeros((1, *bl_data_patch.shape[1:]), dtype=bl_data_patch.dtype, device=bl_data_patch.device)

    fu_point = [fu_point[i] - valid_fu_bbox_lbs[i] for i in range(3)]
    bl_point = [bl_point[i] - valid_bl_bbox_lbs[i] for i in range(3)]

    fu_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(fu_point, shape=fu_data_patch.shape[1:], sigma=1)
    with warnings.catch_warnings():
        # ignore 'The given NumPy array is not writable' warning
        warnings.simplefilter("ignore")
        fu_gauss_point = torch.from_numpy(fu_gauss_point)
    bl_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(bl_point, shape=bl_data_patch.shape[1:], sigma=1)
    with warnings.catch_warnings():
        # ignore 'The given NumPy array is not writable' warning
        warnings.simplefilter("ignore")
        bl_gauss_point = torch.from_numpy(bl_gauss_point)
    fu_padding = [(-min(0, fu_bbox_lbs[i]), max(fu_bbox_ubs[i] - fu_shape[i], 0)) for i in range(3)]
    fu_pad = tuple(v for pair in reversed(fu_padding) for v in pair)
    bl_padding = [(-min(0, bl_bbox_lbs[i]), max(bl_bbox_ubs[i] - bl_shape[i], 0)) for i in range(3)]
    bl_pad = tuple(v for pair in reversed(bl_padding) for v in pair)

    fu_data_patch = F.pad(fu_data_patch, fu_pad, mode="constant", value=0)
    fu_gauss_point = F.pad(fu_gauss_point.unsqueeze(0), fu_pad, mode="constant", value=0)

    bl_data_patch = F.pad(bl_data_patch, bl_pad, mode="constant", value=0)
    bl_seg_patch = F.pad(bl_seg_patch,  bl_pad, mode="constant", value=0)
    bl_gauss_point = F.pad(bl_gauss_point.unsqueeze(0), bl_pad, mode="constant", value=0)

    data = torch.cat((fu_data_patch, bl_data_patch, bl_seg_patch, fu_gauss_point, bl_gauss_point), dim=0).to(torch.device("cuda"))

    predicted_patch = None

    for params in predictor.list_of_parameters:
        if not isinstance(predictor.network, OptimizedModule):
            predictor.network.load_state_dict(params)
        else:
            predictor.network._orig_mod.load_state_dict(params)

        if predicted_patch is None:
            predicted_patch = predictor.predict_sliding_window_return_logits(data)
        else:
            predicted_patch += predictor.predict_sliding_window_return_logits(data)

    predicted_patch = torch.softmax(predicted_patch, dim=0)

    patch_crop_slice = tuple([slice(None, None)] + [slice(fu_padding[i][0], predicted_patch.shape[i+1] - fu_padding[i][1]) for i in range(3)])
    predicted_patch = predicted_patch[patch_crop_slice]
    prediction_slice = tuple([slice(None, None)] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
    prediction = torch.zeros((2, *fu_data.shape[1:]), dtype=predicted_patch.dtype, device=predicted_patch.device)
    prediction[0] = 1
    prediction[1] = 0
    prediction[prediction_slice] = predicted_patch
    return prediction


def predict_patient(images_path: Path, labels_path: Optional[Path], output_path: Path, predictor: LongiSegPredictor,
                    patient: str, tracking_info: dict, dataset_json: dict, mode: str="automatic"):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    all_scans = set()
    if isinstance(tracking_info, dict):
        for l, info in tracking_info.items():
            all_scans.add(info["img_bl"])
            all_scans.add(info["img_fu"])
    elif isinstance(tracking_info, list|tuple):
        for tracking_case in tracking_info:
            for l, info in tracking_case.items():
                all_scans.add(info["img_bl"])
                all_scans.add(info["img_fu"])

    all_data = dict()
    all_segs = dict()
    all_properties = dict()
    for case in all_scans:
        img_paths = [images_path / f"{case}_0000.nii.gz",]

        if labels_path is not None:
            mask_path = labels_path / f"{case}.nii.gz"
            if not mask_path.exists():
                mask_path = None
        else:
            mask_path = None

        data, seg, properties = preprocessor.run_case(img_paths, mask_path, predictor.plans_manager, predictor.configuration_manager,
                                                      dataset_json)
        all_data[case] = torch.from_numpy(data)
        if seg is not None:
            all_segs[case] = torch.from_numpy(seg)
        else:
            all_segs[case] = None
        all_properties[case] = properties

    patch_size = predictor.configuration_manager.patch_size

    if isinstance(tracking_info, dict):
        tracking_info = [tracking_info,]

    for tracking_case in tracking_info:
        for l, info in tracking_case.items():
            bl_point = info["bl_point"]
            if mode == "automatic":
                fu_point = info["fu_point_prop"]
            else:
                fu_point = info["fu_point"]
            if np.isnan(fu_point).all():
                print(f"Patient {patient} lesion {l} has no valid follow-up point, skipping.")
                continue
            bl_point = bl_point[::-1]
            fu_point = fu_point[::-1]
            bl_img = info["img_bl"]
            fu_img = info["img_fu"]
            bl_properties = all_properties[bl_img]
            fu_properties = all_properties[fu_img]
            bl_spacing = [bl_properties['spacing'][i] for i in predictor.plans_manager.transpose_forward]
            fu_spacing = [fu_properties['spacing'][i] for i in predictor.plans_manager.transpose_forward]
            bl_point = [int(bl_point[i] * bl_spacing[i] / predictor.configuration_manager.spacing[i]) for i in range(3)]
            fu_point = [int(fu_point[i] * fu_spacing[i] / predictor.configuration_manager.spacing[i]) for i in range(3)]

            bl_data = all_data[bl_img]
            bl_seg = all_segs[bl_img]
            fu_data = all_data[fu_img]

            try:
                prediction = predict_patch(bl_data, bl_seg, bl_point, l, fu_data, fu_point, predictor, patch_size)
                prediction = predictor.configuration_manager.resampling_fn_probabilities(prediction,
                                                fu_properties['shape_after_cropping_and_before_resampling'],
                                                predictor.configuration_manager.spacing,
                                                [fu_properties['spacing'][i] for i in predictor.plans_manager.transpose_forward])
                prediction = (prediction[1] > 0.5).to(torch.uint8)

                predictor.plans_manager.image_reader_writer_class().write_seg(prediction.cpu().numpy(),
                                                output_path / f"{fu_img}_lesion_{l}{dataset_json['file_ending']}", fu_properties)
            except Exception as e:
                print(f"Prediction for patient {patient}, fu image {fu_img} lesion {l} failed with error: {e}")


def predict(images_path: Path, labels_path: Optional[Path], output_path: Path, model_path: Path, tracking_path: Path, dataset_json_path: Path,
            folds: tuple=(0, 1, 2, 3, 4), disable_tta: bool=True, mode="automatic"):
    tracking_dict = load_json(tracking_path)
    dataset_json = load_json(dataset_json_path)

    predictor = LongiSegPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not disable_tta,
                                    perform_everything_on_device=True, device=torch.device("cuda"), 
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_path, use_folds=folds)

    predictor.configuration_manager.configuration['resampling_fn_data'] = "resample_torch_fornnunet"
    predictor.configuration_manager.configuration['resampling_fn_data_kwargs'] = {
        "is_seg": False,
        "force_separate_z": False,
        "memefficient_seg_resampling": False,
        "device": torch.device("cuda")
    }
    predictor.configuration_manager.configuration['resampling_fn_seg'] = "resample_torch_fornnunet"
    predictor.configuration_manager.configuration['resampling_fn_seg_kwargs'] = {
        "is_seg": True,
        "force_separate_z": False,
        "memefficient_seg_resampling": True,
        "device": torch.device("cuda")
    }
    predictor.configuration_manager.configuration['resampling_fn_probabilities'] = "resample_torch_fornnunet"
    predictor.configuration_manager.configuration['resampling_fn_probabilities_kwargs'] = {
        "is_seg": False,
        "force_separate_z": False,
        "memefficient_seg_resampling": False,
        "device": torch.device("cuda")
    }

    for patient, tracking_info in tqdm(tracking_dict.items(), desc="Predicting patients"):
        predict_patient(images_path, labels_path, output_path, predictor, patient, tracking_info, dataset_json, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=Path, required=True, help="Path to folder containing the input images")
    parser.add_argument("--labels_path", type=Path, required=False, default=None, help="Path to folder containing the input segmentations")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to folder where the predicted segmentations will be stored")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained model folder")
    parser.add_argument("--tracking_path", type=Path, required=True, help="Path to the tracking.json file")
    parser.add_argument("--dataset_json_path", type=Path, required=True, help="Path to the dataset.json file")
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4), help="Folds to use for prediction")
    parser.add_argument("--disable_tta", action="store_true", help="Use this to disable test time augmentation (mirroring)")
    parser.add_argument("--mode", type=str, choices=["automatic", "manual"], default="automatic", help="Whether to use prompts "
                        "obtained through registration (automatic) or 'verified' prompts (manual).")
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    predict(args.images_path, args.labels_path, args.output_path, args.model_path, args.tracking_path, args.dataset_json_path,
            folds=tuple(args.folds), disable_tta=args.disable_tta, mode=args.mode)