import numpy as np
from i3Deep import utils
from tqdm import tqdm
import os


# name = "KGU-53317EB91645"
# load_mask = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/" + name + "/mask.nii.gz"
# load_label_table = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/" + name + "/label_table.txt"
# save_mask = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/" + name + "/mask2.nii.gz"
load_path = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/"


def rename(case_path):
    filenames = utils.load_filenames(case_path + "/", extensions=None)
    for filename in filenames:
        name = os.path.basename(filename)
        if "label" in name and ".nii.gz" in name:
            os.rename(filename, case_path + "/mask.nii.gz")
        elif ".txt" in name:
            os.rename(filename, case_path + "/label_table.txt")
        elif ".nii.gz" in name:
            os.rename(filename, case_path + "/image.nii.gz")


def get_labels(load_label_table):
    with open(load_label_table) as f:
        label_table = f.readlines()
        label_table = np.asarray(label_table)

    ggo = []
    cons = []
    pe = []
    for line in label_table:
        label = line.split()[0]
        if label.isnumeric():
            if "Background" in line or "background" in line:
                continue
            infection = line.split("_")[1]
            keywords = ["ggo", "gg"]
            if any(x in infection.lower() for x in keywords):
                ggo.append(int(label))
            keywords = ["cons", "cns", "con", "cos", "co"]
            if any(x in infection.lower() for x in keywords):
                cons.append(int(label))
            keywords = ["pe", "pes"]
            if any(x in infection.lower() for x in keywords):
                pe.append(int(label))
    return ggo, cons, pe


def merge_labels(load_mask, save_mask, load_label_table):
    mask, affine, spacing, header = utils.load_nifty(load_mask)
    mask = mask.astype(int)
    ggo, cons, pe = get_labels(load_label_table)

    for label in tqdm(np.concatenate((ggo, cons, pe), axis=0), disable=True):
        mask[mask == label] = -label

    for label in tqdm(ggo, disable=True):
        mask[mask == -label] = 1

    for label in tqdm(cons, disable=True):
        mask[mask == -label] = 2

    for label in tqdm(pe, disable=True):
        mask[mask == -label] = 3

    mask = np.rint(mask)
    mask = mask.astype(int)

    utils.save_nifty(save_mask, mask, affine, spacing, header)

def round_mask(filename):
    mask, affine, spacing, header = utils.load_nifty(filename)
    mask = np.rint(mask)
    mask = mask.astype(int)
    utils.save_nifty(filename, mask, affine, spacing, header)

def tmp2(filename):
    mask, affine, spacing, header = utils.load_nifty(filename)
    print(mask[46-1][155-1][116-1])


if __name__ == '__main__':
    # filenames = utils.load_filenames(load_path, extensions=None)
    # for filename in tqdm(filenames):
    #     if os.path.isfile(filename + "/mask2.nii.gz"):
    #         continue
    #     rename(filename)
    #     load_mask = filename + "/mask.nii.gz"
    #     save_mask = filename + "/mask2.nii.gz"
    #     load_label_table = filename + "/label_table.txt"
    #     merge_labels(load_mask, save_mask, load_label_table)

    # for filename in tqdm(filenames):
    #     old_mask = filename + "/mask.nii.gz"
    #     new_mask = filename + "/mask2.nii.gz"
    #     label_table = filename + "/label_table.txt"
    #     if os.path.exists(new_mask):
    #         os.remove(old_mask)
    #         os.rename(new_mask, old_mask)
    #         os.remove(label_table)

    # filenames = utils.load_filenames("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task79_frankfurt3/labelsTr/", extensions=None)
    # for filename in tqdm(filenames):
    #     mask, affine, spacing, header = utils.load_nifty(filename)
    #     mask = np.rint(mask)
    #     mask = mask.astype(np.uint8)
    #     utils.save_nifty(filename, mask, affine, spacing, header)

    # filename = "/gris/gris-f/homelv/kgotkows/datasets/covid19/UK_Frankfurt3/KGU-E9EC0F06F1D6/mask.nii.gz"
    # mask, affine, spacing, header = utils.load_nifty(filename)
    # mask[mask == 5] = 2
    # mask[mask == 6] = 2
    # utils.save_nifty(filename, mask, affine, spacing, header)
    #tmp("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task077_frankfurt3Guided/imagesTr/0001_0001.nii.gz")
    tmp2("/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/Task77_frankfurt3Guided/tmp/900.nii.gz")