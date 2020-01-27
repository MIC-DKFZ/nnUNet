import os
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import numpy as np
from nnunet.utilities.sitk_stuff import copy_geometry


def verify_all_same_orientation(folder):
    """
    This should run after cropping
    :param folder:
    :return:
    """
    nii_files = subfiles(folder, suffix=".nii.gz", join=True)
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    # now we need to check whether they are all the same
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_dataset_integrity(folder, rtol=1e-4, atol=1e-6):
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if all training cases and labels are present
    checks if all test cases (if any) are present
    for each case, checks whether all modalities apre present
    for each case, checks whether the pixel grids are aligned
    :param folder:
    :return:
    """
    assert isfile(join(folder, "dataset.json")), "There needs to be a dataset.json file in folder, folder=%s" % folder
    assert isdir(join(folder, "imagesTr")), "There needs to be a imagesTr subfolder in folder, folder=%s" % folder
    assert isdir(join(folder, "labelsTr")), "There needs to be a labelsTr subfolder in folder, folder=%s" % folder
    dataset = load_json(join(folder, "dataset.json"))
    training_cases = dataset['training']
    num_modalities = len(dataset['modality'].keys())
    test_cases = dataset['test']
    expected_train_identifiers = [i['image'].split("/")[-1][:-7] for i in training_cases]
    expected_test_identifiers = [i.split("/")[-1][:-7] for i in test_cases]

    ## check training set
    nii_files_in_imagesTr = subfiles((join(folder, "imagesTr")), suffix=".nii.gz", join=False)
    nii_files_in_labelsTr = subfiles((join(folder, "labelsTr")), suffix=".nii.gz", join=False)
    # check all cases
    for c in expected_train_identifiers:
        # check if all files are present
        expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
        expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
        assert isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (c, expected_label_file)
        assert all([isfile(i) for i in expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (c, expected_image_files)

        # verify that all modalities and the label have the same shape and geometry.
        label = nib.load(expected_label_file)
        label_shape = label.shape
        label_affine = label.affine
        images = [nib.load(i) for i in expected_image_files]
        image_affines = [i.affine for i in images]
        image_shapes = [i.shape for i in images]
        del images, label
        for i, aff in enumerate(image_affines):
            assert np.all(np.isclose(label_affine, aff, rtol=rtol, atol=atol)), "The affine of the image %s does not match the affine of the label. " \
                                                       "The pixel arrays will not be aligned and nnU-Net cannot use this data. " \
                                                       "Please make sure your image data is coregistered and has the same geometry as the label" % expected_image_files[i]
        for i, sh in enumerate(image_shapes):
            assert np.all(np.isclose(sh, label_shape, rtol=rtol, atol=atol)), "The shape of the image %s does not match the shape of the label. " \
                                                       "Please make sure your image data is coregistered and has the same geometry as the label" % expected_image_files[i]

        # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
        for i in expected_image_files:
            nii_files_in_imagesTr.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    # check for stragglers
    assert len(nii_files_in_imagesTr) == 0, "there are training cases in imagesTr that are not listed in dataset.json: %s" % nii_files_in_imagesTr
    assert len(nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in dataset.json: %s" % nii_files_in_labelsTr

    # check test set, but only if there actually is a test set
    if len(expected_test_identifiers) > 0:
        nii_files_in_imagesTs = subfiles((join(folder, "imagesTs")), suffix=".nii.gz", join=False)

        for c in expected_test_identifiers:
            # check if all files are present
            expected_image_files = [join(folder, "imagesTs", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
            assert all([isfile(i) for i in expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (c, expected_image_files)

            # verify that all modalities and the label have the same geometry. We use the affine for this
            if num_modalities > 1:
                images = [nib.load(i) for i in expected_image_files]
                image_affines = [i.affine for i in images]
                image_shapes = [i.shape for i in images]
                reference_aff = image_affines[0]
                reference_sh = image_shapes[0]
                del images
                for i, aff in enumerate(image_affines[1:]):
                    assert np.all(np.isclose(reference_aff, aff, rtol=rtol, atol=atol)), "The modalities of the image %s do not seem to be registered. Please coregister your modalities." % (expected_image_files[i])
                for i, sh in enumerate(image_shapes[1:]):
                    assert np.all(np.isclose(reference_sh, sh, rtol=rtol, atol=atol)), "The modalities of the image %s do not have the same image shape. Please correct that." % (expected_image_files[i])

            # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
            for i in expected_image_files:
                nii_files_in_imagesTs.remove(os.path.basename(i))
        assert len(nii_files_in_imagesTs) == 0, "there are training cases in imagesTs that are not listed in dataset.json: %s" % nii_files_in_imagesTr

    all_same, unique_orientations = verify_all_same_orientation(join(folder, "imagesTr"))
    if not all_same:
        print("WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you correct that by reorienting the data. fslreorient2std should do the trick")
    # save unique orientations to dataset.json
    print("Dataset OK")


def reorient_to_RAS():
    pass

if __name__ == "__main__":
    # investigate geometry issues
    import SimpleITK as sitk
    # load image
    gt_itk = sitk.ReadImage("/media/fabian/Results/nnUNet/3d_fullres/Task64_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")

    # get numpy array
    pred_npy = sitk.GetArrayFromImage(gt_itk)

    # create new image from numpy array
    prek_itk_new = sitk.GetImageFromArray(pred_npy)
    # copy geometry
    prek_itk_new.CopyInformation(gt_itk)
    #prek_itk_new = copy_geometry(prek_itk_new, gt_itk)

    # save
    sitk.WriteImage(prek_itk_new, "test.mnc")

    # load images in nib
    gt = nib.load("/media/fabian/Results/nnUNet/3d_fullres/Task64_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")
    pred_nib = nib.load("test.mnc")

    new_img_sitk = sitk.ReadImage("test.mnc")

    np1 = sitk.GetArrayFromImage(gt_itk)
    np2 = sitk.GetArrayFromImage(prek_itk_new)
