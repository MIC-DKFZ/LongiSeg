from typing import List, Union
import itertools
import torch
import numpy as np

from acvl_utils.cropping_and_padding.bounding_boxes import int_bbox


# adatped from https://github.com/MIC-DKFZ/acvl_utils/blob/master/acvl_utils/cropping_and_padding/bounding_boxes.py#L357
# to work with None in bboxx
def insert_crop_into_image(
        image: Union[torch.Tensor, np.ndarray],
        crop: Union[torch.Tensor, np.ndarray],
        bbox: List[List[int]]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Inserts a cropped patch back into the original image at the position specified by bbox.
    If the bounding box extends beyond the image boundaries, only the valid portions are inserted.
    If the bounding box lies entirely outside the image, the original image is returned.

    Parameters:
    - image: Original N-dimensional torch.Tensor or np.ndarray to which the crop will be inserted.
    - crop: Cropped patch of the image to be reinserted. May have additional dimensions compared to bbox.
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions of the crop in the original image.

    Returns:
    - image: The original image with the crop reinserted at the specified location (modified in-place).
    """
    # If the bounding box is None and shapes of image and crop are the same, return the crop directly
    if all([b is None for b in itertools.chain(*bbox)]) and image.shape==crop.shape:
        return crop

    # make sure bounding boxes are int and not uint. Otherwise we may get underflow
    bbox = int_bbox(bbox)

    # Ensure that bbox only applies to the last len(bbox) dimensions of crop and image
    num_dims = len(image.shape)
    crop_dims = len(crop.shape)
    bbox_dims = len(bbox)

    if crop_dims < bbox_dims:
        raise ValueError("Bounding box dimensions cannot exceed crop dimensions.")

    # Validate that non-cropped leading dimensions match between image and crop
    leading_dims = num_dims - bbox_dims
    if image.shape[:leading_dims] != crop.shape[:leading_dims]:
        raise ValueError("Leading dimensions of crop and image must match.")

    # Check if the bounding box lies completely outside the image bounds for each cropped dimension
    for i in range(bbox_dims):
        min_val, max_val = bbox[i]
        dim_idx = leading_dims + i  # Corresponding dimension in the image

        if max_val <= 0 or min_val >= image.shape[dim_idx]:
            # If completely out of bounds in any dimension, return the original image
            return image

    # Prepare slices for inserting the crop into the original image
    image_slices = []
    crop_slices = []

    # Iterate over all dimensions, applying bbox only to the last len(bbox) dimensions
    for i in range(num_dims):
        if i < leading_dims:
            # For leading dimensions, use entire dimension (slice(None)) and validate shape
            image_slices.append(slice(None))
            crop_slices.append(slice(None))
        else:
            # For dimensions specified by bbox, calculate the intersection with image bounds
            dim_idx = i - leading_dims
            min_val, max_val = bbox[dim_idx]

            crop_start = max(0, -min_val)  # Start of the crop within the valid area
            image_start = max(0, min_val)  # Start of the image where the crop will be inserted
            image_end = min(max_val, image.shape[i])  # Exclude upper bound by using max_val directly

            # Adjusted range for insertion
            crop_end = crop_start + (image_end - image_start)

            # Append slices for both image and crop insertion ranges
            image_slices.append(slice(image_start, image_end))
            crop_slices.append(slice(crop_start, crop_end))

    # Insert the valid part of the crop back into the original image
    if isinstance(image, torch.Tensor):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    elif isinstance(image, np.ndarray):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    else:
        raise ValueError(f"Unsupported image type {type(image)}")

    return image