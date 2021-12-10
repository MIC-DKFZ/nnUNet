import nrrd
import nibabel as nib
import numpy as np
from i3Deep import utils

load_mask = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/KGU-A8D222271DAD/mask2.nrrd"
save_mask = "D:/Datasets/medical_data/ExportKGU/3D Slicer 2/KGU-A8D222271DAD/mask2.nii.gz"
_nrrd = nrrd.read(load_mask)
data = _nrrd[0]
header = _nrrd[1]
# print(len(_nrrd))
# print(header)

data = utils.reorient(data)

# save nifti
img = nib.Nifti1Image(data, np.eye(4))
nib.save(img, save_mask)