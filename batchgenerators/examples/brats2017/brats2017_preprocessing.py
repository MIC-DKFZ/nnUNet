import numpy as np
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, \
    brats_folder_with_downloaded_train_data, num_threads_for_brats_example
from batchgenerators.utilities.file_and_folder_operations import *

try:
    import SimpleITK as sitk
except ImportError:
    print("You need to have SimpleITK installed to run this example!")
    raise ImportError("SimpleITK not found")

from multiprocessing import Pool


def get_list_of_files(base_dir):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    T1, T1c, T2, FLAIR, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory, p + "_t1.nii.gz")
            t1c_file = join(patient_directory, p + "_t1ce.nii.gz")
            t2_file = join(patient_directory, p + "_t2.nii.gz")
            flair_file = join(patient_directory, p + "_flair.nii.gz")
            seg_file = join(patient_directory, p + "_seg.nii.gz")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), "some file is missing for patient %s; make sure the following " \
                                                        "files are there: %s" % (p, str(this_case))
            list_of_lists.append(this_case)
    print("Found %d patients" % len(list_of_lists))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)

    :param case:
    :param patient_name:
    :return:
    """
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get some metadata
    spacing = imgs_sitk[0].GetSpacing()
    # the spacing returned by SimpleITK is in inverse order relative to the numpy array we receive. If we wanted to
    # resample the data and if the spacing was not isotropic (in BraTS all cases have already been resampled to 1x1x1mm
    # by the organizers) then we need to pay attention here. Therefore we bring the spacing into the correct order so
    # that spacing[0] actually corresponds to the spacing of the first axis of the numpy array
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()

    original_shape = imgs_npy[0].shape

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # now find the nonzero region and crop to that
    nonzero = [np.array(np.where(i != 0)) for i in imgs_npy]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

    # now crop to nonzero
    imgs_npy = imgs_npy[:,
               nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    # now we create a brain mask that we use for normalization
    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy) - 1):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    # the segmentation of brats has the values 0, 1, 2 and 4. This is pretty inconvenient to say the least.
    # We move everything that is 4 to 3
    imgs_npy[-1][imgs_npy[-1] == 4] = 3

    # now save as npz
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    metadata = {
        'spacing': spacing,
        'direction': direction,
        'origin': origin,
        'original_shape': original_shape,
        'nonzero_region': nonzero
    }

    save_pickle(metadata, join(output_folder, patient_name + ".pkl"))


def save_segmentation_as_nifti(segmentation, metadata, output_file):
    original_shape = metadata['original_shape']
    seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
    nonzero = metadata['nonzero_region']
    seg_original_shape[nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1] = segmentation
    sitk_image = sitk.GetImageFromArray(seg_original_shape)
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    # remember to revert spacing back to sitk order again
    sitk_image.SetSpacing(tuple(metadata['spacing'][[2, 1, 0]]))
    sitk.WriteImage(sitk_image, output_file)


if __name__ == "__main__":
    # This is the same preprocessing I used for our contributions to the BraTS 2017 and 2018 challenges.
    # Preprocessing is described in the documentation of load_and_preprocess

    # The training data is identical between BraTS 2017 and 2018. You can request access here:
    # https://ipp.cbica.upenn.edu/#BraTS18_registration

    # brats_base points to where the extracted downloaded training data is

    # preprocessed data is saved as npy. This may seem odd if you are familiar with medical images, but trust me it's
    # the best way to do this for deep learning. It does not make much of a difference for BraTS, but if you are
    # dealing with larger images this is crusial for your pipelines to not get stuck in CPU bottleneck. What we can do
    # with numpy arrays is we can load them via np.load(file, mmap_mode="r") and then read just parts of it on the fly
    # during training. This is super important if your patch size is smaller than the size of the entire patient (for
    # example if you work with large CT data or if you need 2D slices).
    # For this to work properly the output_folder (or wherever the data is stored during training) must be on an SSD!
    # HDDs are usually too slow and you also wouldn't want to do this over a network share (there are exceptions but
    # take this as a rule of thumb)

    # Why is this not an IPython Notebook you may ask? Because I HATE IPython Notebooks. Simple :-)

    list_of_lists = get_list_of_files(brats_folder_with_downloaded_train_data)

    maybe_mkdir_p(brats_preprocessed_folder)

    patient_names = [i[0].split("/")[-2] for i in list_of_lists]

    p = Pool(processes=num_threads_for_brats_example)
    p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [brats_preprocessed_folder] * len(list_of_lists)))
    p.close()
    p.join()

    # remember that we cropped the data before preprocessing. If we predict the test cases, we want to run the same
    # preprocessing for them. We need to then put the segmentation back into its original position (due to cropping).
    # Here is how you can do that:

    # lets use Brats17_2013_0_1 for this example
    img = np.load(join(brats_preprocessed_folder, "Brats17_2013_0_1.npy"))
    metadata = load_pickle(join(brats_preprocessed_folder, "Brats17_2013_0_1.pkl"))
    # remember that we changed the segmentation labels from 0, 1, 2, 4 to 0, 1, 2, 3. We need to change that back to
    # get the correct format
    img[-1][img[-1] == 3] = 4
    save_segmentation_as_nifti(img[-1], metadata, join(brats_preprocessed_folder, "delete_me.nii.gz"))
