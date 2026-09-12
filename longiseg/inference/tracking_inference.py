import argparse
import traceback
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import warnings

import numpy as np
import torch
from torch._dynamo import OptimizedModule
import torch.nn.functional as F

from batchgenerators.utilities.file_and_folder_operations import load_json

from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor
from longiseg.training.dataloading.utils import generated_sparse_to_dense_point_rescaled_gauss
from longiseg.tracking.patch_extraction import compute_paired_patch_bboxes, crop_bbox_to_shape


def _to_torch_pad(padding: List[tuple]) -> tuple:
    return tuple(v for pair in reversed(padding) for v in pair)


def predict_patch(bl_data: torch.Tensor, bl_seg: Optional[torch.Tensor], bl_point: List[int], bl_lesion: int,
                  fu_data: torch.Tensor, fu_point: List[int], predictor: LongiSegPredictor, patch_size: List[int],
                  device: torch.device, sigma: float = 1.0) -> torch.Tensor:
    bl_shape = bl_data.shape[1:]
    fu_shape = fu_data.shape[1:]
    dim = len(fu_shape)

    fu_bbox_lbs, fu_bbox_ubs, bl_bbox_lbs, bl_bbox_ubs = compute_paired_patch_bboxes(fu_shape, fu_point, bl_shape,
                                                                                     bl_point, patch_size)
    valid_fu_bbox_lbs, valid_fu_bbox_ubs, fu_padding = crop_bbox_to_shape(fu_bbox_lbs, fu_bbox_ubs, fu_shape)
    valid_bl_bbox_lbs, valid_bl_bbox_ubs, bl_padding = crop_bbox_to_shape(bl_bbox_lbs, bl_bbox_ubs, bl_shape)

    fu_slice_data = tuple([slice(0, fu_data.shape[0])] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
    fu_data_patch = fu_data[fu_slice_data]

    bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
    bl_data_patch = bl_data[bl_slice_data]

    if bl_seg is not None:
        bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
        bl_seg_patch = bl_seg[bl_slice_seg]
        bl_seg_patch = torch.where(bl_seg_patch == bl_lesion, 1, 0).to(bl_data_patch.dtype)
    else:
        bl_seg_patch = torch.zeros((1, *bl_data_patch.shape[1:]), dtype=bl_data_patch.dtype, device=bl_data_patch.device)

    fu_point = [fu_point[d] - valid_fu_bbox_lbs[d] for d in range(dim)]
    bl_point = [bl_point[d] - valid_bl_bbox_lbs[d] for d in range(dim)]

    fu_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(fu_point, shape=fu_data_patch.shape[1:], sigma=sigma)
    bl_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(bl_point, shape=bl_data_patch.shape[1:], sigma=sigma)
    with warnings.catch_warnings():
        # ignore 'The given NumPy array is not writable' warning
        warnings.simplefilter("ignore")
        fu_gauss_point = torch.from_numpy(fu_gauss_point)
        bl_gauss_point = torch.from_numpy(bl_gauss_point)

    fu_pad = _to_torch_pad(fu_padding)
    bl_pad = _to_torch_pad(bl_padding)

    fu_data_patch = F.pad(fu_data_patch, fu_pad, mode="constant", value=0)
    fu_gauss_point = F.pad(fu_gauss_point.unsqueeze(0), fu_pad, mode="constant", value=0)

    bl_data_patch = F.pad(bl_data_patch, bl_pad, mode="constant", value=0)
    bl_seg_patch = F.pad(bl_seg_patch, bl_pad, mode="constant", value=0)
    bl_gauss_point = F.pad(bl_gauss_point.unsqueeze(0), bl_pad, mode="constant", value=0)

    data = torch.cat((fu_data_patch, bl_data_patch, bl_seg_patch, fu_gauss_point, bl_gauss_point), dim=0).to(device)

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

    patch_crop_slice = tuple([slice(None, None)] + [slice(fu_padding[d][0], predicted_patch.shape[d + 1] - fu_padding[d][1])
                                                    for d in range(dim)])
    predicted_patch = predicted_patch[patch_crop_slice]

    prediction = torch.zeros((2, *fu_shape), dtype=predicted_patch.dtype, device=predicted_patch.device)
    prediction[0] = 1
    prediction_slice = tuple([slice(None, None)] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
    prediction[prediction_slice] = predicted_patch
    return prediction


def predict_patient(images_path: Path, labels_path: Optional[Path], output_path: Path, predictor: LongiSegPredictor,
                    patient: str, tracking_info: dict, dataset_json: dict, device: torch.device,
                    mode: str = "automatic"):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    file_ending = dataset_json['file_ending']
    num_channels = len(dataset_json['channel_names'] if 'channel_names' in dataset_json else dataset_json['modality'])

    if isinstance(tracking_info, dict):
        tracking_info = [tracking_info]

    all_scans = set()
    for tracking_case in tracking_info:
        for info in tracking_case.values():
            all_scans.add(info["img_bl"])
            all_scans.add(info["img_fu"])

    all_data = dict()
    all_segs = dict()
    all_properties = dict()
    for case in all_scans:
        img_paths = [images_path / f"{case}_{c:04d}{file_ending}" for c in range(num_channels)]

        mask_path = labels_path / f"{case}{file_ending}" if labels_path is not None else None
        if mask_path is not None and not mask_path.exists():
            mask_path = None

        data, seg, properties = preprocessor.run_case(img_paths, mask_path, predictor.plans_manager,
                                                      predictor.configuration_manager, dataset_json)
        all_data[case] = torch.from_numpy(data)
        all_segs[case] = torch.from_numpy(seg) if seg is not None else None
        all_properties[case] = properties

    patch_size = predictor.configuration_manager.patch_size
    target_spacing = predictor.configuration_manager.spacing
    transpose_forward = predictor.plans_manager.transpose_forward

    for tracking_case in tracking_info:
        for lesion, info in tracking_case.items():
            fu_point = info["fu_point_prop"] if mode == "automatic" else info["fu_point"]
            if np.isnan(fu_point).all():
                print(f"Patient {patient} lesion {lesion} has no valid follow-up point, skipping.")
                continue

            bl_img = info["img_bl"]
            fu_img = info["img_fu"]
            bl_properties = all_properties[bl_img]
            fu_properties = all_properties[fu_img]
            bl_spacing = [bl_properties['spacing'][i] for i in transpose_forward]
            fu_spacing = [fu_properties['spacing'][i] for i in transpose_forward]
            bl_point = [int(p * bl_spacing[i] / target_spacing[i]) for i, p in enumerate(info["bl_point"][::-1])]
            fu_point = [int(p * fu_spacing[i] / target_spacing[i]) for i, p in enumerate(fu_point[::-1])]

            try:
                prediction = predict_patch(all_data[bl_img], all_segs[bl_img], bl_point, int(lesion),
                                           all_data[fu_img], fu_point, predictor, patch_size, device)
                prediction = predictor.configuration_manager.resampling_fn_probabilities(prediction,
                                                fu_properties['shape_after_cropping_and_before_resampling'],
                                                target_spacing, fu_spacing)
                prediction = (prediction[1] > 0.5).to(torch.uint8)

                predictor.plans_manager.image_reader_writer_class().write_seg(prediction.cpu().numpy(),
                                                output_path / f"{fu_img}_lesion_{lesion}{file_ending}", fu_properties)
            except Exception:
                print(f"Prediction for patient {patient}, fu image {fu_img} lesion {lesion} failed:")
                traceback.print_exc()


def predict(images_path: Path, labels_path: Optional[Path], output_path: Path, model_path: Path, tracking_path: Path,
            dataset_json_path: Path, folds: tuple = (0, 1, 2, 3, 4), disable_tta: bool = True,
            device: torch.device = torch.device("cuda"), mode: str = "automatic"):
    tracking_dict = load_json(tracking_path)
    dataset_json = load_json(dataset_json_path)

    predictor = LongiSegPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not disable_tta,
                                    perform_everything_on_device=True, device=device,
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_path, use_folds=folds)

    for fn, is_seg, memefficient in (('data', False, False), ('seg', True, True), ('probabilities', False, False)):
        predictor.configuration_manager.configuration[f'resampling_fn_{fn}'] = "resample_torch_fornnunet"
        predictor.configuration_manager.configuration[f'resampling_fn_{fn}_kwargs'] = {
            "is_seg": is_seg,
            "force_separate_z": False,
            "memefficient_seg_resampling": memefficient,
            "device": device
        }

    for patient, tracking_info in tqdm(tracking_dict.items(), desc="Predicting patients"):
        predict_patient(images_path, labels_path, output_path, predictor, patient, tracking_info, dataset_json,
                        device, mode=mode)


def tracking_entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=Path, required=True, help="Path to folder containing the input images")
    parser.add_argument("--labels_path", type=Path, required=False, default=None, help="Path to folder containing the input segmentations")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to folder where the predicted segmentations will be stored")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained model folder")
    parser.add_argument("--tracking_path", type=Path, required=True, help="Path to the tracking.json file")
    parser.add_argument("--dataset_json_path", type=Path, required=True, help="Path to the dataset.json file")
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4), help="Folds to use for prediction")
    parser.add_argument("--disable_tta", action="store_true", help="Use this to disable test time augmentation (mirroring)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run inference on")
    parser.add_argument("--mode", type=str, choices=["automatic", "manual"], default="automatic", help="Whether to use prompts "
                        "obtained through registration (automatic) or 'verified' prompts (manual).")
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)

    predict(args.images_path, args.labels_path, args.output_path, args.model_path, args.tracking_path,
            args.dataset_json_path, folds=tuple(args.folds), disable_tta=args.disable_tta,
            device=torch.device(args.device), mode=args.mode)


if __name__ == "__main__":
    tracking_entry_point()
