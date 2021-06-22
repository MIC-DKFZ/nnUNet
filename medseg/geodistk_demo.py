import GeodisTK
import time
import psutil
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image


def geodesic_distance_3d(I, S, spacing, lamb, iter):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)


def demo_geodesic_distance3d():
    image_name = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/imagesTr/0001_0000.nii.gz"
    img = sitk.ReadImage(image_name)
    I = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    I = np.asarray(I, np.float32)

    gt_name = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/imagesTr/0001_0001.nii.gz"
    gt = sitk.ReadImage(gt_name)
    S = sitk.GetArrayFromImage(gt)
    S = np.asarray(S, np.uint8)
    # S = np.zeros_like(I, np.uint8)
    # print(I.shape)
    # S[25][290][350] = 1

    geodesic_distance_map = geodesic_distance_3d(I, S, spacing, 0.99, 4)

    geodesic_distance_map = sitk.GetImageFromArray(geodesic_distance_map)
    geodesic_distance_map.SetSpacing(img.GetSpacing())
    geodesic_distance_map.SetOrigin(img.GetOrigin())
    geodesic_distance_map.SetDirection(img.GetDirection())
    sitk.WriteImage(geodesic_distance_map, "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/geodesic_distance_map.nii.gz")


if __name__ == '__main__':
    demo_geodesic_distance3d()
