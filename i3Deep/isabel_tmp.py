import utils
from os.path import join

load_dir = "/home/k539i/Documents/datasets/corrected_slices"
ids = [str(id).zfill(4) for id in range(96, 129)]

for id in ids:
    axial = utils.load_nifty(join(load_dir, id + "_axial_cor.nii.gz"))[0]
    sagittal = utils.load_nifty(join(load_dir, id + "_sag_cor.nii.gz"))[0]
    coronal = utils.load_nifty(join(load_dir, id + "_cor_cor.nii.gz"))[0]
    print(axial.shape)
    print(sagittal.shape)
    print(coronal.shape)
    break