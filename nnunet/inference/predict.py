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

import argparse
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.experiment_planning.plan_and_preprocess_task import get_caseIDs_from_splitted_dataset_folder
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue
import torch
import SimpleITK as sitk
import shutil
from multiprocessing import Pool

from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.one_hot_encoding import to_one_hot


def predict_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes):
    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(".nii.gz"), "segs_from_prev_stage" \
                                                                                                  " must point to a " \
                                                                                                  "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1, cval=0)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print("error in", l)
            import traceback
            traceback.print_exc()
            errors_in.append(l)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=predict_save_to_queue, args=(trainer.preprocess_patient, q,
                                                         list_of_lists[i::num_processes],
                                                         output_files[i::num_processes],
                                                         segs_from_prev_stage[i::num_processes],
                                                         classes))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate() # this should not happen but better safe than sorry right
            p.join()

        q.close()


def predict_cases(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                  num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True,
                  overwrite_existing=False):

    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

    prman = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds)

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing, segs_from_prev_stage)
    print("starting prediction...")
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)

        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            softmax.append(trainer.predict_preprocessed_data_return_softmax(d, do_tta, 1, False, 1,
                                                                       trainer.data_aug_params['mirror_axes'],
                                                             True, True, 2, trainer.patch_size, True)[None])

        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        """There is a problem with python process communication that prevents us from communicating obejcts 
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
        communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
        filename or np.ndarray and will handle this automatically"""
        if np.prod(softmax_mean.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save
            print("This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        results.append(prman.starmap_async(save_segmentation_nifti_from_softmax,
                                           ((softmax_mean, output_filename, dct, 1, None, None, None, npz_file), )
                                           ))

    _ = [i.get() for i in results]


def predict_from_folder(model, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                                     num_threads_nifti_save, lowres_segmentations, part_id, num_parts, tta,
                        overwrite_existing=True):
    """
    here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases
    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :return:
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    case_ids = get_caseIDs_from_splitted_dataset_folder(input_folder)
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                               "(I was searching for case_id.nii.gz in that folder)"
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None
    return predict_cases(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds, save_npz,
                         num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations,
                         tta, overwrite_existing=overwrite_existing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                           " order (same as training). Files must be named "
                                                           "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                           "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-m', '--model_output_folder', help='model output folder. Will automatically discover the folds '
                                                            'that were '
                                              'run and use those as an ensemble', required=True)
    parser.add_argument('-f', '--folds', nargs='+', default='None', help="folds to use for prediction. Default is None "
                                                                       "which means that folds will be detected "
                                                                       "automatically in the model output folder")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="use this if you want to ensemble"
                                                                                      " these predictions with those of"
                                                                                      " other models. Softmax "
                                                                                      "probabilities will be saved as "
                                                                                      "compresed numpy arrays in "
                                                                                      "output_folder and can be merged "
                                                                                      "between output_folders with "
                                                                                      "merge_predictions.py")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None', help="if model is the highres "
                         "stage of the cascade then you need to use -l to specify where the segmentations of the "
                         "corresponding lowres unet are. Here they are required to do a prediction")
    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_parts", type=int, required=False, default=1, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
                        "Determines many background processes will be used for data preprocessing. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
                        "Determines many background processes will be used for segmentation export. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 2")
    parser.add_argument("--tta", required=False, type=int, default=1, help="Set to 0 to disable test time data "
                                                                           "augmentation (speedup of factor "
                                                                           "4(2D)/8(3D)), "
                                                                           "lower quality segmentations")
    parser.add_argument("--overwrite_existing", required=False, type=int, default=1, help="Set this to 0 if you need "
                                                                                          "to resume a previous "
                                                                                          "prediction. Default: 1 "
                                                                                          "(=existing segmentations "
                                                                                          "in output_folder will be "
                                                                                          "overwritten)")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    model = args.model_output_folder
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    tta = args.tta
    overwrite = args.overwrite_existing

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if overwrite == 0:
        overwrite = False
    elif overwrite == 1:
        overwrite = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    predict_from_folder(model, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, tta,
                        overwrite_existing=overwrite)

