from i3Deep import utils
import numpy as np
from tqdm import tqdm
import os
import json
import copy
import shutil
from pathlib import Path
import SimpleITK as sitk

image_path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/imagesTs1/"
kgu_image_path = "/gris/gris-f/homelv/kgotkows/datasets/covid19/UK_Frankfurt3_dataset/images/"

image_filenames = utils.load_filenames(image_path)
kgu_image_filenames = utils.load_filenames(kgu_image_path)

mapping = []
kgu_shapes = {}

for kgu_image_filename in tqdm(kgu_image_filenames, desc="KGU"):
    kgu_name = os.path.basename(kgu_image_filename)[:-7]
    # kgu_image, _, _, _ = utils.load_nifty(kgu_image_filename)
    # kgu_shapes[kgu_name] = kgu_image.shape
    # Set up the reader and get the file information
    reader = sitk.ImageFileReader()
    reader.SetFileName(kgu_image_filename)  # Give it the mha file as a string
    reader.LoadPrivateTagsOn()  # Make sure it can get all the info
    reader.ReadImageInformation()  # Get just the information from the file
    shape = reader.GetSize()  # If you want the x, y, z
    kgu_shapes[kgu_name] = shape

for image_filename in tqdm(image_filenames, desc="Image"):
    name = os.path.basename(image_filename)[:-7]
    image, affine, spacing, header = utils.load_nifty(image_filename)
    found = False
    for kgu_image_filename in tqdm(kgu_image_filenames, desc="KGU"):
        kgu_name = os.path.basename(kgu_image_filename)[:-7]
        if image.shape == kgu_shapes[kgu_name]:
            kgu_image, _, _, _ = utils.load_nifty(kgu_image_filename)

            if np.allclose(image, kgu_image, atol=5):
                mapping.append({"Name": name, "KGU": kgu_name})
                print("{} = {}".format(name, kgu_name))
                kgu_image_filenames.remove(kgu_image_filename)
                found = True
                break
    if not found:
        raise RuntimeError("No mapping found!")
print(mapping)
