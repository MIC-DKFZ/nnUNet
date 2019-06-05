#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from multiprocessing.pool import Pool

import numpy as np
from nnunet.evaluation.evaluator import aggregate_scores
from scipy.ndimage import label
import SimpleITK as sitk
from nnunet.utilities.sitk_stuff import copy_geometry
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


def load_remove_save(input_file: str, output_file: str, for_which_classes: list):
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)

    img_out_itk = sitk.GetImageFromArray(remove_all_but_the_largest_connected_component(img_npy, for_which_classes))
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes

    for c in for_which_classes:
        mask = image == c
        lmap, num_objects = label(mask.astype(int))
        if num_objects > 1:
            sizes = []
            for o in range(1, num_objects + 1):
                sizes.append((lmap == o).sum())
            mx = np.argmax(sizes) + 1
            image[(lmap != mx) & mask] = 0
    return image


def consolidate_folds(output_folder_base):
    folds = list(range(5))
    folders_folds = [join(output_folder_base, "fold_%d" % i) for i in folds]

    assert all([isdir(i) for i in folders_folds]), "some folds are missing"

    # now for each fold, read the postprocessing json
    postprocessing_jsons = [load_json(join(output_folder_base, "fold_%d" % f, "postprocessing.json")) for f in folds]
    validation_raw_folders = [join(output_folder_base, "fold_%d" % i, postprocessing_jsons[i]['validation_raw']) for i in folds]

    # count niftis in there
    num_niftis = 0
    for v in validation_raw_folders:
        num_niftis += len(subfiles(v, suffix=".nii.gz"))

    num_niftis_gt = len(subfiles(join(output_folder_base, "gt_niftis")))

    assert num_niftis == num_niftis_gt, "some fodls are missing predicted niftis :-(. Make sure you ran all folds properly"

    # first we need to know what classes we have
    classes = np.array([int(i) for i in postprocessing_jsons[0]['dc_per_class_raw'].keys()])

    arr_raw = np.zeros((len(folds), len(classes)))
    arr_pp = np.zeros((len(folds), len(classes)))
    num_samples = []
    for f in folds:
        tmp = postprocessing_jsons[f]
        for i, c in enumerate(classes):
            arr_raw[f, i] = tmp['dc_per_class_raw'][str(c)]
            arr_pp[f, i] = tmp['dc_per_class_pp'][str(c)]
        num_samples.append(tmp['num_samples'])
    num_samples = np.array(num_samples)[:, None]

    arr_raw *= num_samples
    arr_pp *= num_samples

    arr_raw = arr_raw.sum(0) / sum(num_samples)
    arr_pp = arr_pp.sum(0) / sum(num_samples)

    for_which_classes = []
    for i in range(len(arr_pp)):
        if arr_pp[i] > arr_raw[i]:
            c = classes[i]
            if c != 0:
                for_which_classes.append(int(c))

    out = {'for_which_classes': for_which_classes,
           'classes': [int(i) for i in classes],
           'dc_raw': list(arr_raw),
           'dc_pp': list(arr_pp)}

    save_json(out, join(output_folder_base, "postprocessing_consolidated.json"))

    # now we need to apply the consolidated postprocessing to the niftis from the cross-validation and evaluate that
    output_folder = join(output_folder_base, "cv_niftis_postprocessed")
    output_folder_raw = join(output_folder_base, "cv_niftis_raw")
    maybe_mkdir_p(output_folder)
    maybe_mkdir_p(output_folder_raw)
    p = Pool(8)
    results = []
    for f in folds:
        niftis = subfiles(validation_raw_folders[f], suffix=".nii.gz")
        for n in niftis:
            n_f = n.split("/")[-1]
            output_file = join(output_folder, n_f)
            shutil.copy(join(n), join(output_folder_raw))
            results.append(p.starmap_async(load_remove_save, ((n, output_file, out['for_which_classes']),)))
    _ = [i.get() for i in results]
    p.close()
    p.join()

    pred_gt_tuples = [(join(output_folder_base, "cv_niftis_postprocessed", f),
                       join(output_folder_base, "gt_niftis", f))
                      for f in subfiles(join(output_folder_base, "gt_niftis"), suffix=".nii.gz", join=False)]

    aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(output_folder_base, "cv_niftis_postprocessed", "summary.json"),
                         json_author="Fabian", num_threads=8)

    pred_gt_tuples = [(join(output_folder_base, "cv_niftis_raw", f),
                       join(output_folder_base, "gt_niftis", f))
                      for f in subfiles(join(output_folder_base, "gt_niftis"), suffix=".nii.gz", join=False)]

    aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(output_folder_base, "cv_niftis_raw", "summary.json"),
                         json_author="Fabian", num_threads=8)


def determine_postprocessing(base, gt_labels_folder, raw_subfolder_name="validation_raw",
                             test_pp_subf_name="test_postprocess__validation",
                             final_subf_name="validation_final", delete_temp=True, processes=8):
    """

    :param base:
    :param gt_labels_folder:
    :param raw_subfolder_name:
    :param test_pp_subf_name:
    :param final_subf_name:
    :param delete_temp:
    :param processes:
    :return:
    """
    maybe_mkdir_p(join(base, test_pp_subf_name))
    maybe_mkdir_p(join(base, final_subf_name))

    p = Pool(processes)

    pred_gt_tuples = []
    results = []
    print("generating dummy postprocessed data")

    assert isfile(join(base, raw_subfolder_name, "summary.json"))

    # lets see what classes are in the dataset
    classes = [int(i) for i in load_json(join(base, raw_subfolder_name, "summary.json"))['results']['mean'].keys() if int(i) != 0]

    fnames = subfiles(join(base, raw_subfolder_name), suffix=".nii.gz", join=False)
    # now determine postprocessing
    for f in fnames:
        predicted_segmentation = join(base, raw_subfolder_name, f)
        # now remove all but the largest connected component for each class
        output_file = join(base, test_pp_subf_name, f)
        results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, classes),)))
        pred_gt_tuples.append([output_file, join(gt_labels_folder, f)])

    _ = [i.get() for i in results]

    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(base, test_pp_subf_name, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    # now we need to load both the evaluation before and after postprocessing and then decide for each class the
    # result was better
    print("determining which postprocessing to use...")

    pp_results = {}
    pp_results['dc_per_class_raw'] = {}
    pp_results['dc_per_class_pp'] = {}
    pp_results['for_which_classes'] = []

    validation_result_raw = load_json(join(base, raw_subfolder_name, "summary.json"))['results']
    pp_results['num_samples'] = len(validation_result_raw['all'])

    validation_result_PP_test = load_json(join(base, test_pp_subf_name, "summary.json"))['results']['mean']

    validation_result_raw = validation_result_raw['mean']

    for c in classes:
        dc_raw = validation_result_raw[str(c)]['Dice']
        dc_pp = validation_result_PP_test[str(c)]['Dice']
        pp_results['dc_per_class_raw'][str(c)] = dc_raw
        pp_results['dc_per_class_pp'][str(c)] = dc_pp

        if c != 0 and dc_pp > dc_raw:
            pp_results['for_which_classes'].append(int(c))

    pp_results['validation_raw'] = raw_subfolder_name
    pp_results['validation_final'] = final_subf_name

    save_json(pp_results, join(base, "postprocessing.json"))

    print("done. for_which_classes: ", pp_results['for_which_classes'])

    # now that we have a proper for_which_classes, apply that
    print("applying that to prediction...")

    pred_gt_tuples = []
    results = []
    for f in fnames:
        predicted_segmentation = join(base, raw_subfolder_name, f)

        # now remove all but the largest connected component for each class
        output_file = join(base, final_subf_name, f)
        results.append(p.starmap_async(load_remove_save, (
        (predicted_segmentation, output_file, pp_results['for_which_classes']),)))

        pred_gt_tuples.append([output_file,
                               join(gt_labels_folder, f)])

    _ = [i.get() for i in results]
    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(base, final_subf_name, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    if delete_temp:
        shutil.rmtree(join(base, test_pp_subf_name))

    p.close()
    p.join()
    print("done")

