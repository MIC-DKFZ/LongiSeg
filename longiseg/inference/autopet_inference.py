from pathlib import Path
from tqdm import tqdm
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import torch
from torch._dynamo import OptimizedModule
import torch.nn.functional as F

from batchgenerators.utilities.file_and_folder_operations import load_json, join

import longiseg
from longiseg.preprocessing.resampling.default_resampling import compute_new_shape
from longiseg.utilities.find_class_by_name import recursive_find_python_class
from longiseg.utilities.plans_handling.plans_handler import PlansManager

from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO
from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor
from longiseg.training.dataloading.utils import generated_sparse_to_dense_point_rescaled_gauss


def preprocess_data(data, seg, properties, plans_manager, configuration_manager, uniques=None):
    data = data.astype(np.float32)  # this creates a copy
    data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

    has_seg = seg is not None
    if has_seg:
        seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

    target_spacing = configuration_manager.spacing  # this should already be transposed

    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    if not has_seg:
        seg = np.zeros((1, *data.shape[1:]), dtype=np.uint8)
    data = _normalize(data, seg, configuration_manager,
                            plans_manager.foreground_intensity_properties_per_channel)

    with warnings.catch_warnings():
        # ignore 'The given NumPy array is not writable' warning
        warnings.simplefilter("ignore")
        data = torch.from_numpy(data).to(torch.device("cuda"))
    data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)

    if has_seg:
        with warnings.catch_warnings():
            # ignore 'The given NumPy array is not writable' warning
            warnings.simplefilter("ignore")
            seg = torch.from_numpy(seg).to(torch.device("cuda"))
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing, uniques=uniques)    
    else:
        seg = None
    return data, seg, properties


def _normalize(data, seg, configuration_manager, foreground_intensity_properties_per_channel):
    for c in range(data.shape[0]):
        scheme = configuration_manager.normalization_schemes[c]
        normalizer_class = recursive_find_python_class(join(longiseg.__path__[0], "preprocessing", "normalization"),
                                                        scheme,
                                                        'longiseg.preprocessing.normalization')
        if normalizer_class is None:
            raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
        normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                        intensityproperties=foreground_intensity_properties_per_channel[str(c)])
        data[c] = normalizer.run(data[c], seg[0])
    return data


def predict_case(input_dict: dict, model_path: Path, folds: tuple=(0, 1, 2, 3, 4), disable_tta: bool=True) -> sitk.Image:
    primary_bl_image_path = input_dict.get("primary_bl_image_path", None)
    primary_bl_image_path = input_dict.get("primary_bl_image_path", None)
    primary_fu_image_path = input_dict.get("primary_fu_image_path", None)
    primary_bl_mask_path = input_dict.get("primary_bl_mask_path", None)
    secondary_bl_image_path = input_dict.get("secondary_bl_image_path", None)
    secondary_fu_image_path = input_dict.get("secondary_fu_image_path", None)
    secondary_bl_mask_path = input_dict.get("secondary_bl_mask_path", None)
    primary_bl_clickpoints = input_dict.get("primary_bl_clickpoints", None)
    primary_fu_clickpoints = input_dict.get("primary_fu_clickpoints", None)
    secondary_bl_clickpoints = input_dict.get("secondary_bl_clickpoints", None)
    secondary_fu_clickpoints = input_dict.get("secondary_fu_clickpoints", None)

    plans = load_json(model_path / "plans.json")
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration("3d_fullres")

    configuration_manager.configuration['resampling_fn_data'] = "resample_torch_forautopet"
    configuration_manager.configuration['resampling_fn_data_kwargs'] = {
        "is_seg": False,
        "force_separate_z": False,
        "memefficient_seg_resampling": False,
        "device": torch.device("cuda")
    }
    configuration_manager.configuration['resampling_fn_seg'] = "resample_torch_forautopet"
    configuration_manager.configuration['resampling_fn_seg_kwargs'] = {
        "is_seg": True,
        "force_separate_z": False,
        "memefficient_seg_resampling": True,
        "device": torch.device("cuda")
    }
    configuration_manager.configuration['resampling_fn_probabilities'] = "resample_torch_forautopet"
    configuration_manager.configuration['resampling_fn_probabilities_kwargs'] = {
        "is_seg": False,
        "force_separate_z": False,
        "memefficient_seg_resampling": False,
        "device": torch.device("cuda")
    }

    target_spacing = configuration_manager.spacing
    patch_size = configuration_manager.patch_size

    tracking = dict()

    if primary_bl_image_path is not None and primary_bl_mask_path is not None:
        primary_bl_image, primary_bl_properties = SimpleITKIO().read_images([primary_bl_image_path,])
        print(f"Primary baseline image shape: {primary_bl_image.shape}")
        print(f"Primary baseline image spacing: {primary_bl_properties['spacing']}")
        primary_bl_mask, _ = SimpleITKIO().read_seg(primary_bl_mask_path)
        primary_bl_spacing = [primary_bl_properties['spacing'][i] for i in plans_manager.transpose_forward]
        for l, point in primary_bl_clickpoints.items():
            if l not in tracking:
                tracking[l] = {}
            point = point[::-1]
            point = point = [point[i] for i in plans_manager.transpose_forward]
            point = [int(p * primary_bl_spacing[i] / target_spacing[i])
                            for i, p in enumerate(point)]
            tracking[l]["baseline"] = {"scan": "primary", "point": point}
        primary_bl_image, primary_bl_mask, primary_bl_properties = \
            preprocess_data(primary_bl_image, primary_bl_mask, primary_bl_properties, plans_manager,
                           configuration_manager, uniques=[0] + list(primary_bl_clickpoints.keys()))
        primary_bl_image = primary_bl_image.to(torch.device("cpu"))
        primary_bl_mask = primary_bl_mask.to(torch.device("cpu"))
    else:
        primary_bl_image, primary_bl_properties, primary_bl_mask = None, None, None
    if primary_fu_image_path is not None:
        primary_fu_image, primary_fu_properties = SimpleITKIO().read_images([primary_fu_image_path,])
        print(f"Primary follow-up image shape: {primary_fu_image.shape}")
        print(f"Primary follow-up image spacing: {primary_fu_properties['spacing']}")
        primary_fu_spacing = [primary_fu_properties['spacing'][i] for i in plans_manager.transpose_forward]
        primary_fu_properties['shape_after_cropping_and_before_resampling'] = primary_fu_image.shape[1:]
        for l, point in primary_fu_clickpoints.items():
            if l not in tracking:
                tracking[l] = {}
            point = point[::-1]
            point = [point[i] for i in plans_manager.transpose_forward]
            point = [int(p * primary_fu_spacing[i] / target_spacing[i])
                            for i, p in enumerate(point)]
            tracking[l]["followup"] = {"scan": "primary", "point": point}
        primary_fu_image, _, primary_fu_properties = \
            preprocess_data(primary_fu_image, None, primary_fu_properties, plans_manager,
                           configuration_manager, uniques=[0] + list(primary_fu_clickpoints.keys()))
        primary_fu_image = primary_fu_image.to(torch.device("cpu"))
    else:
        primary_fu_image, primary_fu_properties = None, None
    if secondary_bl_image_path is not None and secondary_bl_mask_path is not None:
        secondary_bl_image, secondary_bl_properties = SimpleITKIO().read_images([secondary_bl_image_path,])
        print(f"Secondary baseline image shape: {secondary_bl_image.shape}")
        print(f"Secondary baseline image spacing: {secondary_bl_properties['spacing']}")
        secondary_bl_mask, _ = SimpleITKIO().read_seg(secondary_bl_mask_path)
        secondary_bl_spacing = [secondary_bl_properties['spacing'][i] for i in plans_manager.transpose_forward]
        for l, point in secondary_bl_clickpoints.items():
            if l not in tracking:
                tracking[l] = {}
            point = point[::-1]
            point = [point[i] for i in plans_manager.transpose_forward]
            point = [int(p * secondary_bl_spacing[i] / target_spacing[i])
                            for i, p in enumerate(point)]
            tracking[l]["baseline"] = {"scan": "secondary", "point": point}
        secondary_bl_image, secondary_bl_mask, secondary_bl_properties = \
            preprocess_data(secondary_bl_image, secondary_bl_mask, secondary_bl_properties, plans_manager,
                           configuration_manager, uniques=[0] + list(secondary_bl_clickpoints.keys()))
        secondary_bl_image = secondary_bl_image.to(torch.device("cpu"))
        secondary_bl_mask = secondary_bl_mask.to(torch.device("cpu"))
    else:
        secondary_bl_image, secondary_bl_properties, secondary_bl_mask = None, None, None
    if secondary_fu_image_path is not None:
        secondary_fu_image, secondary_fu_properties = SimpleITKIO().read_images([secondary_fu_image_path,])
        print(f"Secondary follow-up image shape: {secondary_fu_image.shape}")
        print(f"Secondary follow-up image spacing: {secondary_fu_properties['spacing']}")
        secondary_fu_spacing = [secondary_fu_properties['spacing'][i] for i in plans_manager.transpose_forward]
        secondary_fu_properties['shape_after_cropping_and_before_resampling'] = secondary_fu_image.shape[1:]
        for l, point in secondary_fu_clickpoints.items():
            if l not in tracking:
                tracking[l] = {}
            point = point[::-1]
            point = [point[i] for i in plans_manager.transpose_forward]
            point = [int(p * secondary_fu_spacing[i] / target_spacing[i])
                            for i, p in enumerate(point)]
            tracking[l]["followup"] = {"scan": "secondary", "point": point}
        secondary_fu_image, _, secondary_fu_properties = \
            preprocess_data(secondary_fu_image, None, secondary_fu_properties, plans_manager,
                           configuration_manager, uniques=[0] + list(secondary_fu_clickpoints.keys()))
        secondary_fu_image = secondary_fu_image.to(torch.device("cpu"))
    else:
        secondary_fu_image, secondary_fu_properties = None, None

    if primary_bl_image is not None:
        primary_bl_image = primary_bl_image.to(torch.device("cuda"))
        primary_bl_mask = primary_bl_mask.to(torch.device("cuda"))
    if primary_fu_image is not None:
        primary_fu_image = primary_fu_image.to(torch.device("cuda"))
    if secondary_bl_image is not None:
        secondary_bl_image = secondary_bl_image.to(torch.device("cuda"))
        secondary_bl_mask = secondary_bl_mask.to(torch.device("cuda"))
    if secondary_fu_image is not None:
        secondary_fu_image = secondary_fu_image.to(torch.device("cuda"))

    predictor = LongiSegPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not disable_tta,
                                    perform_everything_on_device=True, device=torch.device("cuda"), 
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_path, use_folds=folds)

    patch_dict = dict()

    for l in tqdm(list(tracking.keys()), desc="Predicting lesions"):
        if not "baseline" in tracking[l].keys() or not "followup" in tracking[l].keys():
            del tracking[l]
            continue
        bl_scan = tracking[l]["baseline"]["scan"]
        bl_data = primary_bl_image if bl_scan == "primary" else secondary_bl_image
        bl_seg = primary_bl_mask if bl_scan == "primary" else secondary_bl_mask
        fu_scan = tracking[l]["followup"]["scan"]
        fu_data = primary_fu_image if fu_scan == "primary" else secondary_fu_image
        bl_point = tracking[l]["baseline"]["point"]
        fu_point = tracking[l]["followup"]["point"]

        fu_shape = fu_data.shape[1:]
        bl_shape = bl_data.shape[1:]

        if not fu_scan in patch_dict.keys():
            patch_dict[fu_scan] = {"shape": fu_shape, "lesions": {}}

        fu_bbox_lbs = []
        fu_bbox_ubs = []
        bl_bbox_lbs = []
        bl_bbox_ubs = []

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
                                    f"patch_size {patch_size}, data_shape {fu_shape} "
                                    f"for patient {p}")
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
        bl_slice_data = tuple([slice(0, bl_data.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])
        bl_slice_seg = tuple([slice(0, bl_seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bl_bbox_lbs, valid_bl_bbox_ubs)])

        fu_data = fu_data[fu_slice_data]
        bl_data = bl_data[bl_slice_data]
        bl_seg = bl_seg[bl_slice_seg]

        bl_seg = torch.where(bl_seg == l, 1, 0).to(fu_data.dtype)

        fu_point = [fu_point[i] - valid_fu_bbox_lbs[i] for i in range(3)]
        bl_point = [bl_point[i] - valid_bl_bbox_lbs[i] for i in range(3)]

        try:
            fu_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(fu_point, shape=fu_data.shape[1:], sigma=1)
        except Exception as e:
            print(f"Couldn't generate Gaussian blob for follow up point, probably because the point is outside the image.\n"
                  f"Skipping lesion {l}. Full error:\n{e}")
            continue
        with warnings.catch_warnings():
            # ignore 'The given NumPy array is not writable' warning
            warnings.simplefilter("ignore")
            fu_gauss_point = torch.from_numpy(fu_gauss_point).to(torch.device("cuda"))
        try:
            bl_gauss_point = generated_sparse_to_dense_point_rescaled_gauss(bl_point, shape=bl_data.shape[1:], sigma=1)
        except Exception as e:
            print(f"Couldn't generate Gaussian blob for baseline point, probably because the point is outside the image.\n"
                  f"Skipping lesion {l}. Full error:\n{e}")
            continue
        with warnings.catch_warnings():
            # ignore 'The given NumPy array is not writable' warning
            warnings.simplefilter("ignore")
            bl_gauss_point = torch.from_numpy(bl_gauss_point).to(torch.device("cuda"))

        fu_padding = [(-min(0, fu_bbox_lbs[i]), max(fu_bbox_ubs[i] - fu_shape[i], 0)) for i in range(3)]
        fu_pad = tuple(v for pair in reversed(fu_padding) for v in pair)
        bl_padding = [(-min(0, bl_bbox_lbs[i]), max(bl_bbox_ubs[i] - bl_shape[i], 0)) for i in range(3)]
        bl_pad = tuple(v for pair in reversed(bl_padding) for v in pair)

        fu_data = F.pad(fu_data, fu_pad, mode="constant", value=0)
        fu_gauss_point = F.pad(fu_gauss_point.unsqueeze(0), fu_pad, mode="constant", value=0)

        bl_data = F.pad(bl_data, bl_pad, mode="constant", value=0)
        bl_seg = F.pad(bl_seg,  bl_pad, mode="constant", value=0)
        bl_gauss_point = F.pad(bl_gauss_point.unsqueeze(0), bl_pad, mode="constant", value=0)

        data = torch.cat((fu_data, bl_data, bl_seg, fu_gauss_point, bl_gauss_point), dim=0)

        predicted_patch = None

        for params in predictor.list_of_parameters:
            if not isinstance(predictor.network, OptimizedModule):
                predictor.network.load_state_dict(params)
            else:
                predictor.network._orig_mod.load_state_dict(params)

            if predicted_patch is None:
                predicted_patch = predictor.predict_sliding_window_return_logits(data).to('cpu')
            else:
                predicted_patch += predictor.predict_sliding_window_return_logits(data).to('cpu')

        predicted_patch = torch.softmax(predicted_patch, dim=0).to(torch.device("cuda"))

        patch_crop_slice = tuple([slice(None, None)] + [slice(fu_padding[i][0], predicted_patch.shape[i+1] - fu_padding[i][1]) for i in range(3)])
        predicted_patch = predicted_patch[patch_crop_slice]
        prediction_slice = tuple([slice(None, None)] + [slice(i, j) for i, j in zip(valid_fu_bbox_lbs, valid_fu_bbox_ubs)])
        patch_dict[fu_scan]["lesions"][l] = {
            "patch": predicted_patch,
            "slice": prediction_slice
        }

    prediction_dict = dict()
    for k, v in patch_dict.items():
        properties_dict = primary_fu_properties if k == "primary" else secondary_fu_properties
        full_prediction = None
        for l, p in tqdm(list(v["lesions"].items()), desc=f"Exporting predictions for {k}"):
            prediction_slice = tuple([slice(None, None), *p["slice"][1:]])
            lesion_prediction = torch.zeros((2, *v["shape"]), dtype=torch.float32, device=torch.device("cuda"))
            lesion_prediction[0] = 1
            lesion_prediction[1] = 0
            lesion_prediction[prediction_slice] = p["patch"]
            lesion_prediction = configuration_manager.resampling_fn_probabilities(lesion_prediction,
                                                    properties_dict['shape_after_cropping_and_before_resampling'],
                                                    configuration_manager.spacing,
                                                    [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
            lesion_prediction = (lesion_prediction[1] > 0.5).to(torch.uint8)*int(l)
            if full_prediction is None:
                full_prediction = lesion_prediction
            else:
                full_prediction = torch.maximum(full_prediction, lesion_prediction)
        itk_image = sitk.GetImageFromArray(full_prediction.cpu().numpy())
        itk_image.SetSpacing(properties_dict['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties_dict['sitk_stuff']['origin'])
        itk_image.SetDirection(properties_dict['sitk_stuff']['direction'])
        prediction_dict[k] = itk_image
    return prediction_dict