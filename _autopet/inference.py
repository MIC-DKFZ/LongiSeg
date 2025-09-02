from pathlib import Path

import SimpleITK as sitk
import json
import os

import torch

from longiseg.inference.autopet_inference import predict_case


IN_PATH = Path("/input/")
OUT_PATH = Path("/output/")
MODEL_PATH = Path("_model")


def parse_clickpoints_from_json(json_paths):
    def get_clickpoints(files):
        clickpoints = {}
        for bl_file in files:
            with open(bl_file, 'r') as f:
                data = json.load(f)
                
            for _, point in enumerate(data.get("points", [])):
                point_name = point.get("name", "")
                # Extract lesion_id from point name (format: "Lesion X")
                lesion_id = int(point_name.split()[-1])
                    
                # Extract coordinates
                coords = point.get("point", [])
                if len(coords) == 3:
                    clickpoints[lesion_id] = tuple(coords)
        return clickpoints if clickpoints else None
    
    try:
        primary_bl_files = [p for p in json_paths if "primary-baseline" in p.name.lower()]
        primary_fu_files = [p for p in json_paths if "primary-followup" in p.name.lower()]
        secondary_bl_files = [p for p in json_paths if "secondary-baseline" in p.name.lower()]
        secondary_fu_files = [p for p in json_paths if "secondary-followup" in p.name.lower()]
        
        # Process all files and return tuple of results
        return (
            get_clickpoints(primary_bl_files),
            get_clickpoints(primary_fu_files),
            get_clickpoints(secondary_bl_files),
            get_clickpoints(secondary_fu_files)
        )
        
    except Exception as e:
        print(f"Error parsing JSON files: {str(e)}")
        return None, None, None, None


def get_input():
    json_files = [IN_PATH / f for f in os.listdir(IN_PATH) if f.endswith('.json')]

    images= {}
    images_dir = IN_PATH / "images"
    for subdir in images_dir.glob("*"):
            if subdir.is_dir():
                subfolder_name = subdir.name
                # Find all image files in this subdirectory
                for ext in ["*.mha", "*.nii", "*.nii.gz"]:
                    for img_path in subdir.glob(ext):
                        images[subfolder_name] = img_path
                        break 
    primary_bl_clicks, primary_fu_clicks, secondary_bl_clicks, secondary_fu_clicks = parse_clickpoints_from_json(json_files)
    all_inputs = {
                    "primary_bl_image_path": images.get('primary-baseline-ct', None),
                    "primary_fu_image_path": images.get('primary-followup-ct', None),
                    "primary_bl_mask_path": images.get('primary-baseline-ct-tumor-lesion-seg', None),
                    "secondary_bl_image_path": images.get('secondary-baseline-ct', None),
                    "secondary_fu_image_path": images.get('secondary-followup-ct', None),
                    "secondary_bl_mask_path": images.get('secondary-baseline-ct-tumor-lesion-seg', None),
                    "primary_bl_clickpoints": primary_bl_clicks,
                    "primary_fu_clickpoints": primary_fu_clicks,
                    "secondary_bl_clickpoints": secondary_bl_clicks,
                    "secondary_fu_clickpoints": secondary_fu_clicks,
                }
    return all_inputs


def save_output(out_dict):
    output_dir = OUT_PATH / "images"

    os.makedirs(output_dir / "primary-followup-ct-tumor-lesion-seg", exist_ok=True)
    sitk.WriteImage(out_dict["primary"], str(output_dir / "primary-followup-ct-tumor-lesion-seg" / "segmentation.mha"))
    if out_dict.get("secondary", None) is not None:
        os.makedirs(output_dir / "secondary-followup-ct-tumor-lesion-seg", exist_ok=True)
        sitk.WriteImage(out_dict["secondary"], str(output_dir / "secondary-followup-ct-tumor-lesion-seg" / "segmentation.mha"))


def run():
    _show_torch_cuda_info()

    all_inputs = get_input()

    ### Actual prediction here ###

    out_dict = predict_case(all_inputs, model_path=MODEL_PATH)

    save_output(out_dict)


def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(torch.__version__)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
