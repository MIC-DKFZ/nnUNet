import SimpleITK as sitk
import numpy as np
import glob

file_list = glob.glob(r"C:\Users\Test\Desktop\Bart\Data\Dataset CT Lymph Nodes\labels\MEDLN_053.nii.gz")

for file in file_list:
    img = sitk.ReadImage(file)
    data = sitk.GetArrayFromImage(img)
    labels = np.unique(data)
    print(f"{file}: {labels}")
