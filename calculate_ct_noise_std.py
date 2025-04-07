import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import argparse

import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

def calculate_noise_and_mean_hu_in_roi(ct_image_path, segmentation_mask_path, organ_label, roi_size=(3, 32, 32)):
    """
    Load a CT image and its corresponding segmentation mask, extract a fixed ROI around the centroid of the specified organ,
    and calculate both the noise (standard deviation) and mean Hounsfield Unit (HU) within the ROI, considering only the
    pixels within the organ's mask.

    Parameters:
    - ct_image_path (str): Path to the CT image file.
    - segmentation_mask_path (str): Path to the segmentation mask file.
    - organ_label (int): Label value of the organ to analyze.
    - roi_size (tuple, optional): Size of the ROI to extract around the centroid in voxels (Z, Y, X). Defaults to (32, 32, 32).

    Returns:
    - tuple: (noise, mean_hu)
        - noise (float): Standard deviation of HU within the ROI and organ mask.
        - mean_hu (float): Mean HU within the ROI and organ mask.

    Raises:
    - ValueError: If organ_label is not found in the segmentation mask, or if the ROI extraction fails.
    """
    try:
        # Load CT image
        ct_image = sitk.ReadImage(ct_image_path)
        ct_array = sitk.GetArrayFromImage(ct_image)  # Shape: (Z, Y, X)

        # Load segmentation mask
        seg_mask = sitk.ReadImage(segmentation_mask_path)
        seg_array = sitk.GetArrayFromImage(seg_mask)  # Shape: (Z, Y, X)

        # Validate that shapes match
        if ct_array.shape != seg_array.shape:
            raise ValueError("CT image and segmentation mask shapes do not match.")

        # Create binary mask for the specified organ
        organ_mask = (seg_array == organ_label)
        if not np.any(organ_mask):
            raise ValueError(f"Organ label {organ_label} not found in the segmentation mask.")

        # Calculate the centroid of the organ in voxel coordinates
        centroid_voxel = center_of_mass(organ_mask)
        if np.isnan(centroid_voxel).any():
            raise ValueError("Centroid calculation resulted in NaN. Check the segmentation mask.")

        # Convert centroid to integer voxel indices
        centroid_voxel = tuple(map(int, np.round(centroid_voxel)))

        # Define the bounding box around the centroid
        half_size = tuple(s // 2 for s in roi_size)
        z_min = max(centroid_voxel[0] - half_size[0], 0)
        y_min = max(centroid_voxel[1] - half_size[1], 0)
        x_min = max(centroid_voxel[2] - half_size[2], 0)

        z_max = min(centroid_voxel[0] + half_size[0], ct_array.shape[0])
        y_max = min(centroid_voxel[1] + half_size[1], ct_array.shape[1])
        x_max = min(centroid_voxel[2] + half_size[2], ct_array.shape[2])

        # Extract the sub-volume and sub-mask
        sub_volume = ct_array[z_min:z_max, y_min:y_max, x_min:x_max]
        sub_mask = organ_mask[z_min:z_max, y_min:y_max, x_min:x_max]

        if sub_volume.size == 0:
            raise ValueError("Extracted ROI has zero size. Check the ROI size and organ location.")

        # Apply the mask to the sub-volume to get only organ pixels
        organ_pixels = sub_volume[sub_mask]

        if organ_pixels.size == 0:
            raise ValueError("No organ pixels found within the extracted ROI. Check the segmentation mask and ROI parameters.")

        # Calculate noise (standard deviation) and mean HU within the ROI
        noise = np.std(organ_pixels)
        mean_hu = np.mean(organ_pixels)

        return noise, mean_hu

    except Exception as e:
        print(f"Error in calculate_noise_and_mean_hu_in_roi: {e}")
        return np.nan, np.nan

def main():
    """
    Main function to execute the noise calculation script.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate CT noise (standard deviation) around a specified organ.")
    parser.add_argument('--ct', type=str, required=True, help="Path to the CT image file (e.g., NIfTI, DICOM).")
    parser.add_argument('--seg', type=str, required=True, help="Path to the segmentation mask file.")
    parser.add_argument('--label', type=int, required=True, help="Organ label to extract from the segmentation mask.")
    parser.add_argument('--size', type=int, nargs=3, default=(32, 32, 32), help="ROI size around centroid (Z Y X). Defaults to 32 32 32.")

    args = parser.parse_args()

    # Extract arguments
    ct_image_path = args.ct
    segmentation_mask_path = args.seg
    organ_label = args.label
    roi_size = tuple(args.size)

    # Validate file paths
    if not os.path.isfile(ct_image_path):
        print(f"CT image file not found: {ct_image_path}")
        return
    if not os.path.isfile(segmentation_mask_path):
        print(f"Segmentation mask file not found: {segmentation_mask_path}")
        return

    print("Starting noise calculation...")
    noise = calculate_noise_in_roi(ct_image_path, segmentation_mask_path, organ_label, roi_size)

    if not np.isnan(noise):
        print(f"Noise (Standard Deviation) around organ label {organ_label}: {noise:.2f}")
    else:
        print("Noise calculation failed.")

if __name__ == "__main__":
    main()