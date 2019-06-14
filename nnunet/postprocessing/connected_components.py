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
from copy import deepcopy
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
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes

    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
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
    # TODO just assume validation folder names, skip dependency on pp.json files
    """
    you must have run the validation from nnUNetV2, otherwise postprocessing.json files will be missing
    :param output_folder_base:
    :return:
    """
    folds = list(range(5))
    folders_folds = [join(output_folder_base, "fold_%d" % i) for i in folds]

    assert all([isdir(i) for i in folders_folds]), "some folds are missing"

    # now for each fold, read the postprocessing json. this will tell us what the name of the validation folder is
    postprocessing_jsons = [load_json(join(output_folder_base, "fold_%d" % f, "postprocessing.json")) for f in folds]
    validation_raw_folders = [join(output_folder_base, "fold_%d" % i, postprocessing_jsons[i]['validation_raw']) for i in folds]

    # count niftis in there
    num_niftis = 0
    for v in validation_raw_folders:
        num_niftis += len(subfiles(v, suffix=".nii.gz"))

    num_niftis_gt = len(subfiles(join(output_folder_base, "gt_niftis")))

    assert num_niftis == num_niftis_gt, "some folds are missing predicted niftis :-(. Make sure you ran all folds properly"

    # now we need to apply the consolidated postprocessing to the niftis from the cross-validation and evaluate that
    output_folder_raw = join(output_folder_base, "cv_niftis_raw")
    maybe_mkdir_p(output_folder_raw)
    for f in folds:
        niftis = subfiles(validation_raw_folders[f], suffix=".nii.gz")
        for n in niftis:
            shutil.copy(n, join(output_folder_raw))

    determine_postprocessing(output_folder_base, join(output_folder_base, "gt_niftis"), 'cv_niftis_raw',
                             final_subf_name="cv_niftis_postprocessed", processes=8)


def load_for_which_classes(pkl_file):
    '''
    loads the relevant part of the pkl file that is needed for applying postprocessing
    :param pkl_file:
    :return:
    '''
    a = load_pickle(pkl_file)
    return a['for_which_classes']


def determine_postprocessing(base, gt_labels_folder, raw_subfolder_name="validation_raw",
                             temp_folder="temp",
                             final_subf_name="validation_final", processes=8):
    """
    :param base:
    :param gt_labels_folder:
    :param raw_subfolder_name:
    :param temp_folder: used to store temporary data, will be deleted after we are done here
    :param final_subf_name:
    :param processes:
    :return:
    """
    # lets see what classes are in the dataset
    classes = [int(i) for i in load_json(join(base, raw_subfolder_name, "summary.json"))['results']['mean'].keys() if int(i) != 0]

    # multiprocessing rules
    p = Pool(processes)
    results = []  # used to collect mp 'results'. Not really a result

    assert isfile(join(base, raw_subfolder_name, "summary.json")), "join(base, raw_subfolder_name) does not " \
                                                                   "contain a summary.json"

    # these are all the files we will be dealing with
    fnames = subfiles(join(base, raw_subfolder_name), suffix=".nii.gz", join=False)

    # make output and temp dir
    maybe_mkdir_p(join(base, temp_folder))
    maybe_mkdir_p(join(base, final_subf_name))

    pp_results = {}
    pp_results['dc_per_class_raw'] = {}
    pp_results['dc_per_class_pp_all'] = {}  # dice scores after treating all foreground classes as one
    pp_results['dc_per_class_pp_per_class'] = {}  # dice scores after removing everything except larges cc
    # independently for each class after we already did dc_per_class_pp_all
    pp_results['for_which_classes'] = []

    validation_result_raw = load_json(join(base, raw_subfolder_name, "summary.json"))['results']
    pp_results['num_samples'] = len(validation_result_raw['all'])
    validation_result_raw = validation_result_raw['mean']

    pred_gt_tuples = []

    # first treat all foreground classes as one and remove all but the largest foreground connected component
    for f in fnames:
        predicted_segmentation = join(base, raw_subfolder_name, f)
        # now remove all but the largest connected component for each class
        output_file = join(base, temp_folder, f)
        results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, (classes, )),)))
        pred_gt_tuples.append([output_file, join(gt_labels_folder, f)])

    _ = [i.get() for i in results]

    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(base, temp_folder, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    # now we need to figure out if doing this improved the dice scores. We will implement that defensively in so far
    # that if a single class got worse as a result we won't do this. We can change this in the future but right now I
    # prefer to do it this way
    validation_result_PP_test = load_json(join(base, temp_folder, "summary.json"))['results']['mean']

    for c in classes:
        dc_raw = validation_result_raw[str(c)]['Dice']
        dc_pp = validation_result_PP_test[str(c)]['Dice']
        pp_results['dc_per_class_raw'][str(c)] = dc_raw
        pp_results['dc_per_class_pp_all'][str(c)] = dc_pp

    # true if new is better
    do_fg_cc = False
    comp = [pp_results['dc_per_class_pp_all'][str(cl)] > pp_results['dc_per_class_raw'][str(cl)] for cl in classes]
    if any(comp):
        # at least one class improved - yay!
        # now check if another got worse
        # true if new is worse
        any_worse = any([pp_results['dc_per_class_pp_all'][str(cl)] < pp_results['dc_per_class_raw'][str(cl)] for cl in classes])
        if not any_worse:
            pp_results['for_which_classes'].append(classes)
            do_fg_cc = True
    else:
        # did not improve things - don't do it
        pass

    # now depending on whether we do remove all but the largest foreground connected component we define the source dir
    # for the next one to be the raw or the temp dir
    if do_fg_cc:
        source = join(base, temp_folder)
    else:
        source = join(base, raw_subfolder_name)

    pred_gt_tuples = []
    for f in fnames:
        predicted_segmentation = join(source, f)
        output_file = join(base, temp_folder, f)
        results.append(p.starmap_async(load_remove_save, ((predicted_segmentation, output_file, classes),)))
        pred_gt_tuples.append([output_file, join(gt_labels_folder, f)])

    _ = [i.get() for i in results]

    # evaluate postprocessed predictions
    _ = aggregate_scores(pred_gt_tuples, labels=classes,
                         json_output_file=join(base, temp_folder, "summary.json"),
                         json_author="Fabian", num_threads=processes)

    if do_fg_cc:
        old_res = deepcopy(validation_result_PP_test)
    else:
        old_res = validation_result_raw

    # these are the new dice scores
    validation_result_PP_test = load_json(join(base, temp_folder, "summary.json"))['results']['mean']

    for c in classes:
        dc_raw = old_res[str(c)]['Dice']
        dc_pp = validation_result_PP_test[str(c)]['Dice']
        pp_results['dc_per_class_pp_per_class'][str(c)] = dc_pp

        if dc_pp > dc_raw:
            pp_results['for_which_classes'].append(int(c))

    pp_results['validation_raw'] = raw_subfolder_name
    pp_results['validation_final'] = final_subf_name

    save_json(pp_results, join(base, "postprocessing.json"))

    # now that we have a proper for_which_classes, apply that
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

    # delete temp
    shutil.rmtree(join(base, temp_folder))

    p.close()
    p.join()
    print("done")

