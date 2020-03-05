import os
from batchgenerators.utilities.file_and_folder_operations import *
import nibabel as nib
import numpy as np
from nnunet.utilities.sitk_stuff import copy_geometry
import SimpleITK as sitk
from nnunet.configuration import default_num_threads
from multiprocessing import Pool


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


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()

    same_ori = np.all(np.isclose(ori1, ori2))
    if not same_ori: print("the origin does not match between the images")
    same_spac = np.all(np.isclose(spacing1, spacing2))
    if not same_spac: print("the spacing does not match between the images")
    same_dir = np.all(np.isclose(direction1, direction2))
    if not same_dir: print("the direction does not match between the images")
    same_size = np.all(np.isclose(size1, size2))
    if not same_size: print("the size does not match between the images")

    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False


def verify_contains_only_expected_labels(itk_img: str, valid_labels: (tuple, list)):
    img_npy = sitk.GetArrayFromImage(sitk.ReadImage(itk_img))
    uniques = np.unique(img_npy)
    invalid_uniques = [i for i in uniques if i not in valid_labels]
    if len(invalid_uniques) == 0:
        r = True
    else:
        r = False
    return r, invalid_uniques


def verify_dataset_integrity(folder):
    """
    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if all training cases and labels are present
    checks if all test cases (if any) are present
    for each case, checks whether all modalities apre present
    for each case, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should
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

    label_files = []

    # check all cases
    print("Verifying training set")
    for c in expected_train_identifiers:
        print("checking case", c)
        # check if all files are present
        expected_label_file = join(folder, "labelsTr", c + ".nii.gz")
        label_files.append(expected_label_file)
        expected_image_files = [join(folder, "imagesTr", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
        assert isfile(expected_label_file), "could not find label file for case %s. Expected file: \n%s" % (
        c, expected_label_file)
        assert all([isfile(i) for i in
                    expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
        c, expected_image_files)

        # verify that all modalities and the label have the same shape and geometry.
        label_itk = sitk.ReadImage(expected_label_file)
        images_itk = [sitk.ReadImage(i) for i in expected_image_files]
        for i, img in enumerate(images_itk):
            same_geometry = verify_same_geometry(img, label_itk)
            assert same_geometry, "The geometry of the image %s does not match the geometry of the label file. " \
                                  "The pixel arrays will not be aligned and nnU-Net cannot use this data. " \
                                  "Please make sure your image modalities are coregistered and have the same geometry " \
                                  "as the label" % expected_image_files[i]

        # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
        for i in expected_image_files:
            nii_files_in_imagesTr.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    # check for stragglers
    assert len(
        nii_files_in_imagesTr) == 0, "there are training cases in imagesTr that are not listed in dataset.json: %s" % nii_files_in_imagesTr
    assert len(
        nii_files_in_labelsTr) == 0, "there are training cases in labelsTr that are not listed in dataset.json: %s" % nii_files_in_labelsTr

    # verify that only properly declared values are present in the labels
    print("Verifying label values")
    expected_labels = list(int(i) for i in dataset['labels'].keys())
    p = Pool(default_num_threads)
    results = p.starmap(verify_contains_only_expected_labels, zip(label_files, [expected_labels] * len(label_files)))
    p.close()
    p.join()

    fail = False
    print("Expected label values are", expected_labels)
    for i, r in enumerate(results):
        if not r[0]:
            print("Unexpected labels found in file %s. Found these unexpected values (they should not be there) %s" % (
            label_files[i], r[1]))
            fail = True

    if fail:
        raise AssertionError(
            "Found unexpected labels in the training dataset. Please correct that or adjust your dataset.json accordingly")
    else:
        print("Labels OK")

    # check test set, but only if there actually is a test set
    if len(expected_test_identifiers) > 0:
        print("Verifying test set")
        nii_files_in_imagesTs = subfiles((join(folder, "imagesTs")), suffix=".nii.gz", join=False)

        for c in expected_test_identifiers:
            # check if all files are present
            expected_image_files = [join(folder, "imagesTs", c + "_%04.0d.nii.gz" % i) for i in range(num_modalities)]
            assert all([isfile(i) for i in
                        expected_image_files]), "some image files are missing for case %s. Expected files:\n %s" % (
            c, expected_image_files)

            # verify that all modalities and the label have the same geometry. We use the affine for this
            if num_modalities > 1:
                images_itk = [sitk.ReadImage(i) for i in expected_image_files]
                reference_img = images_itk[0]

                for i, img in enumerate(images_itk[1:]):
                    assert verify_same_geometry(img, reference_img), "The modalities of the image %s do not seem to be " \
                                                                     "registered. Please coregister your modalities." % (
                                                                     expected_image_files[i])

            # now remove checked files from the lists nii_files_in_imagesTr and nii_files_in_labelsTr
            for i in expected_image_files:
                nii_files_in_imagesTs.remove(os.path.basename(i))
        assert len(
            nii_files_in_imagesTs) == 0, "there are training cases in imagesTs that are not listed in dataset.json: %s" % nii_files_in_imagesTr

    all_same, unique_orientations = verify_all_same_orientation(join(folder, "imagesTr"))
    if not all_same:
        print(
            "WARNING: Not all images in the dataset have the same axis ordering. We very strongly recommend you correct that by reorienting the data. fslreorient2std should do the trick")
    # save unique orientations to dataset.json
    print("Dataset OK")


def reorient_to_RAS(img_fname: str, output_fname: str = None):
    img = nib.load(img_fname)
    canonical_img = nib.as_closest_canonical(img)
    if output_fname is None:
        output_fname = img_fname
    nib.save(canonical_img, output_fname)


if __name__ == "__main__":
    # investigate geometry issues
    import SimpleITK as sitk

    # load image
    gt_itk = sitk.ReadImage(
        "/media/fabian/Results/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")

    # get numpy array
    pred_npy = sitk.GetArrayFromImage(gt_itk)

    # create new image from numpy array
    prek_itk_new = sitk.GetImageFromArray(pred_npy)
    # copy geometry
    prek_itk_new.CopyInformation(gt_itk)
    # prek_itk_new = copy_geometry(prek_itk_new, gt_itk)

    # save
    sitk.WriteImage(prek_itk_new, "test.mnc")

    # load images in nib
    gt = nib.load(
        "/media/fabian/Results/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")
    pred_nib = nib.load("test.mnc")

    new_img_sitk = sitk.ReadImage("test.mnc")

    np1 = sitk.GetArrayFromImage(gt_itk)
    np2 = sitk.GetArrayFromImage(prek_itk_new)
