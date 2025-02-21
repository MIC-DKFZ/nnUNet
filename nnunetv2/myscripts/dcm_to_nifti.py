import os
import numpy as np
import pydicom
import nibabel as nib

def convert_dicom_to_nifti(dicom_dir, output_dir):
    """
    Convert DICOM files in a directory to a single NIfTI (.nii.gz) file using DICOM metadata.

    Parameters:
    - dicom_dir (str): Path to the directory containing DICOM files.
    - output_dir (str): Path to the directory where the NIfTI file will be saved.
    """
    # Get all DICOM files in the directory
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    if not dicom_files:
        raise ValueError("No DICOM files found in the specified directory.")

    # Read and sort DICOM slices by InstanceNumber
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda s: int(s.InstanceNumber))

    # Extract pixel data and metadata
    pixel_arrays = [s.pixel_array for s in slices]
    volume = np.stack(pixel_arrays, axis=-1)  # Create 3D volume

    # Extract voxel spacing
    pixel_spacing = [float(val) for val in slices[0].PixelSpacing]
    try:
        slice_thickness = float(slices[0].SliceThickness)
    except AttributeError:
        # Fallback: calculate slice thickness from ImagePositionPatient
        z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
        slice_thickness = np.abs(np.diff(z_positions).mean())

    voxel_spacing = (*pixel_spacing, slice_thickness)

    # Extract affine transformation
    orientation = [float(val) for val in slices[0].ImageOrientationPatient]
    position = [float(val) for val in slices[0].ImagePositionPatient]
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal_cosine = np.cross(row_cosine, col_cosine)

    affine = np.eye(4)
    affine[:3, 0] = np.array(row_cosine) * voxel_spacing[0]
    affine[:3, 1] = np.array(col_cosine) * voxel_spacing[1]
    affine[:3, 2] = np.array(normal_cosine) * voxel_spacing[2]
    affine[:3, 3] = position

    # Create and save NIfTI file
    os.makedirs(output_dir, exist_ok=True)
    nifti_img = nib.Nifti1Image(volume, affine)
    output_file = os.path.join(output_dir, "converted_image.nii.gz")
    nib.save(nifti_img, output_file)
    print(f"NIfTI file saved to: {output_file}")



def convert_segmentation_dicom_to_nifti(dicom_dir, output_dir):
    """
    Convert a single stacked DICOM file (segmentation) to a NIfTI file.

    Parameters:
    - dicom_file (str): Path to the DICOM segmentation file.
    - output_dir (str): Path to the directory where the NIfTI file will be saved.
    """

    dicom_file = os.path.join(dicom_dir, os.listdir(dicom_dir)[0])
    # Load the DICOM file
    seg_dcm = pydicom.dcmread(dicom_file)

    # Extract pixel data
    pixel_data = seg_dcm.pixel_array  # Already a 3D volume
    if seg_dcm.get((0x0062, 0x0001)) == "BINARY":  # Segmentation type tag
        pixel_data = pixel_data.astype(np.uint8)  # Ensure binary segmentation

    # Get metadata using tags
    pixel_spacing = [float(val) for val in seg_dcm.get((0x0028, 0x0030), [1.0, 1.0])]  # Pixel Spacing
    slice_thickness = float(seg_dcm.get((0x0018, 0x0050), 1.0))  # Slice Thickness
    voxel_spacing = (*pixel_spacing, slice_thickness)

    orientation = [float(val) for val in seg_dcm.get((0x0020, 0x0037), [1.0] * 6)]  # Image Orientation
    position = [float(val) for val in seg_dcm.get((0x0020, 0x0032), [0.0, 0.0, 0.0])]  # Image Position

    # Compute affine matrix
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    normal_cosine = np.cross(row_cosine, col_cosine)

    affine = np.eye(4)
    affine[:3, 0] = np.array(row_cosine) * voxel_spacing[0]
    affine[:3, 1] = np.array(col_cosine) * voxel_spacing[1]
    affine[:3, 2] = np.array(normal_cosine) * voxel_spacing[2]
    affine[:3, 3] = position

    # Save the NIfTI file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "segmentation.nii.gz")
    nifti_img = nib.Nifti1Image(pixel_data, affine)
    nib.save(nifti_img, output_file)
    print(f"NIfTI file saved to: {output_file}")


# Example usage:
dicom_directory = "C:/Users/Test/Desktop/Bart/Data/dcmtonifti/09-14-2014-MEDLYMPH001-mediastinallymphnodes-78073/300.000000-Lymph node segmentations-96296"
output_directory = "C:/Users/Test/Desktop/Bart/Data/dcmtonifti/09-14-2014-MEDLYMPH001-mediastinallymphnodes-78073/300.000000-Lymph node segmentations-96296"
# convert_dicom_to_nifti(dicom_directory, output_directory)
convert_segmentation_dicom_to_nifti(dicom_directory, output_directory)



filepath = os.path.join(dicom_directory, os.listdir(dicom_directory)[0])
ds = pydicom.filereader.dcmread(filepath)
print(ds)
