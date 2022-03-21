#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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


import zipfile
from multiprocessing.pool import Pool

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.file_and_folder_operations_winos import * # Join path by slash on windows system.
import shutil
from nnunet.paths import default_cascade_trainer, default_plans_identifier, default_trainer, network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from subprocess import call


def copy_fold(in_folder: str, out_folder: str):
    shutil.copy(join(in_folder, "debug.json"), join(out_folder, "debug.json"))
    shutil.copy(join(in_folder, "model_final_checkpoint.model"), join(out_folder, "model_final_checkpoint.model"))
    shutil.copy(join(in_folder, "model_final_checkpoint.model.pkl"),
                join(out_folder, "model_final_checkpoint.model.pkl"))
    shutil.copy(join(in_folder, "progress.png"), join(out_folder, "progress.png"))
    if isfile(join(in_folder, "network_architecture.pdf")):
        shutil.copy(join(in_folder, "network_architecture.pdf"), join(out_folder, "network_architecture.pdf"))


def copy_model(directory: str, output_directory: str):
    """

    :param directory: must have the 5 fold_X subfolders as well as a postprocessing.json and plans.pkl
    :param output_directory:
    :return:
    """
    expected_folders = ["fold_%d" % i for i in range(5)]
    assert all([isdir(join(directory, i)) for i in expected_folders]), "not all folds present"

    assert isfile(join(directory, "plans.pkl")), "plans.pkl missing"
    assert isfile(join(directory, "postprocessing.json")), "postprocessing.json missing"

    for e in expected_folders:
        maybe_mkdir_p(join(output_directory, e))
        copy_fold(join(directory, e), join(output_directory, e))

    shutil.copy(join(directory, "plans.pkl"), join(output_directory, "plans.pkl"))
    shutil.copy(join(directory, "postprocessing.json"), join(output_directory, "postprocessing.json"))


def copy_pretrained_models_for_task(task_name: str, output_directory: str,
                                    models: tuple = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                                    nnunet_trainer=default_trainer,
                                    nnunet_trainer_cascade=default_cascade_trainer,
                                    plans_identifier=default_plans_identifier):
    trainer_output_dir = nnunet_trainer + "__" + plans_identifier
    trainer_output_dir_cascade = nnunet_trainer_cascade + "__" + plans_identifier

    for m in models:
        to = trainer_output_dir_cascade if m == "3d_cascade_fullres" else trainer_output_dir
        expected_output_folder = join(network_training_output_dir, m, task_name, to)
        if not isdir(expected_output_folder):
            if m == "3d_lowres" or m == "3d_cascade_fullres":
                print("Task", task_name, "does not seem to have the cascade")
                continue
            else:
                raise RuntimeError("missing folder! %s" % expected_output_folder)
        output_here = join(output_directory, m, task_name, to)
        maybe_mkdir_p(output_here)
        copy_model(expected_output_folder, output_here)


def check_if_valid(ensemble: str, valid_models, valid_trainers, valid_plans):
    ensemble = ensemble[len("ensemble_"):]
    mb1, mb2 = ensemble.split("--")
    c1, tr1, p1 = mb1.split("__")
    c2, tr2, p2 = mb2.split("__")
    if c1 not in valid_models: return False
    if c2 not in valid_models: return False
    if tr1 not in valid_trainers: return False
    if tr2 not in valid_trainers: return False
    if p1 not in valid_plans: return False
    if p2 not in valid_plans: return False
    return True


def copy_ensembles(taskname, output_folder, valid_models=('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'),
                   valid_trainers=(default_trainer, default_cascade_trainer),
                   valid_plans=(default_plans_identifier,)):
    ensemble_dir = join(network_training_output_dir, 'ensembles', taskname)
    if not isdir(ensemble_dir):
        print("No ensemble directory found for task", taskname)
        return
    subd = subdirs(ensemble_dir, join=False)
    valid = []
    for s in subd:
        v = check_if_valid(s, valid_models, valid_trainers, valid_plans)
        if v:
            valid.append(s)
    output_ensemble = join(output_folder, 'ensembles', taskname)
    maybe_mkdir_p(output_ensemble)
    for v in valid:
        this_output = join(output_ensemble, v)
        maybe_mkdir_p(this_output)
        shutil.copy(join(ensemble_dir, v, 'postprocessing.json'), this_output)


def compress_everything(output_base, num_processes=8):
    p = Pool(num_processes)
    tasks = subfolders(output_base, join=False)
    tasknames = [i.split('/')[-1] for i in tasks]
    args = []
    for t, tn in zip(tasks, tasknames):
        args.append((join(output_base, tn + ".zip"), join(output_base, t)))
    p.starmap(compress_folder, args)
    p.close()
    p.join()


def compress_folder(zip_file, folder):
    """inspired by https://stackoverflow.com/questions/1855095/how-to-create-a-zip-archive-of-a-directory-in-python"""
    zipf = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            zipf.write(join(root, file), os.path.relpath(join(root, file), folder))


def export_one_task(taskname, models, output_folder, nnunet_trainer=default_trainer,
                    nnunet_trainer_cascade=default_cascade_trainer,
                    plans_identifier=default_plans_identifier):
    copy_pretrained_models_for_task(taskname, output_folder, models, nnunet_trainer, nnunet_trainer_cascade,
                                    plans_identifier)
    copy_ensembles(taskname, output_folder, models, (nnunet_trainer, nnunet_trainer_cascade), (plans_identifier,))
    compress_folder(join(output_folder, taskname + '.zip'), join(output_folder, taskname))


def export_pretrained_model(task_name: str, output_file: str,
                            models: tuple = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"),
                            nnunet_trainer=default_trainer,
                            nnunet_trainer_cascade=default_cascade_trainer,
                            plans_identifier=default_plans_identifier,
                            folds=(0, 1, 2, 3, 4), strict=True):
    zipf = zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)
    trainer_output_dir = nnunet_trainer + "__" + plans_identifier
    trainer_output_dir_cascade = nnunet_trainer_cascade + "__" + plans_identifier

    for m in models:
        to = trainer_output_dir_cascade if m == "3d_cascade_fullres" else trainer_output_dir
        expected_output_folder = join(network_training_output_dir, m, task_name, to)
        if not isdir(expected_output_folder):
            if strict:
                raise RuntimeError("Task %s is missing the model %s" % (task_name, m))
            else:
                continue

        expected_folders = ["fold_%d" % i if i != 'all' else i for i in folds]
        assert all([isdir(join(expected_output_folder, i)) for i in expected_folders]), "not all requested folds " \
                                                                                        "present, " \
                                                                                        "Task %s model %s" % \
                                                                                        (task_name, m)

        assert isfile(join(expected_output_folder, "plans.pkl")), "plans.pkl missing, Task %s model %s" % (task_name, m)

        for e in expected_folders:
            zipf.write(join(expected_output_folder, e, "debug.json"),
                       os.path.relpath(join(expected_output_folder, e, "debug.json"),
                                       network_training_output_dir))
            zipf.write(join(expected_output_folder, e, "model_final_checkpoint.model"),
                       os.path.relpath(join(expected_output_folder, e, "model_final_checkpoint.model"),
                                       network_training_output_dir))
            zipf.write(join(expected_output_folder, e, "model_final_checkpoint.model.pkl"),
                       os.path.relpath(join(expected_output_folder, e, "model_final_checkpoint.model.pkl"),
                                       network_training_output_dir))
            zipf.write(join(expected_output_folder, e, "progress.png"),
                       os.path.relpath(join(expected_output_folder, e, "progress.png"), network_training_output_dir))
            if isfile(join(expected_output_folder, e, "network_architecture.pdf")):
                zipf.write(join(expected_output_folder, e, "network_architecture.pdf"),
                           os.path.relpath(join(expected_output_folder, e, "network_architecture.pdf"),
                                           network_training_output_dir))

        zipf.write(join(expected_output_folder, "plans.pkl"),
                   os.path.relpath(join(expected_output_folder, "plans.pkl"), network_training_output_dir))
        if not isfile(join(expected_output_folder, "postprocessing.json")):
            if strict:
                raise RuntimeError('postprocessing.json missing. Run nnUNet_determine_postprocessing or disable strict')
            else:
                print('WARNING: postprocessing.json missing')
        else:
            zipf.write(join(expected_output_folder, "postprocessing.json"),
                       os.path.relpath(join(expected_output_folder, "postprocessing.json"), network_training_output_dir))

    ensemble_dir = join(network_training_output_dir, 'ensembles', task_name)
    if not isdir(ensemble_dir):
        print("No ensemble directory found for task", task_name)
        return
    subd = subdirs(ensemble_dir, join=False)
    valid = []
    for s in subd:
        v = check_if_valid(s, models, (nnunet_trainer, nnunet_trainer_cascade), (plans_identifier))
        if v:
            valid.append(s)
    for v in valid:
        zipf.write(join(ensemble_dir, v, 'postprocessing.json'),
                   os.path.relpath(join(ensemble_dir, v, 'postprocessing.json'),
                                   network_training_output_dir))
    zipf.close()


def export_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description="Use this script to export models to a zip file for sharing with "
                                                 "others. You can upload the zip file and then either share the url "
                                                 "for usage with nnUNet_download_pretrained_model_by_url, or share the "
                                                 "zip for usage with nnUNet_install_pretrained_model_from_zip")
    parser.add_argument('-t', type=str, help='task name or task id')
    parser.add_argument('-o', type=str, help='output file name. Should end with .zip')
    parser.add_argument('-m', nargs='+',
                        help='list of model configurations. Default: 2d 3d_lowres 3d_fullres 3d_cascade_fullres. Must '
                             'be adapted to fit the available models of a task',
                        default=("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"), required=False)
    parser.add_argument('-tr', type=str, help='trainer class used for 2d 3d_lowres and 3d_fullres. '
                                              'Default: %s' % default_trainer, required=False, default=default_trainer)
    parser.add_argument('-trc', type=str, help='trainer class used for 3d_cascade_fullres. '
                                              'Default: %s' % default_cascade_trainer, required=False,
                        default=default_cascade_trainer)
    parser.add_argument('-pl', type=str, help='nnunet plans identifier. Default: %s' % default_plans_identifier,
                        required=False, default=default_plans_identifier)
    parser.add_argument('--disable_strict', action='store_true', help='set this if you want to allow skipping '
                                                                     'missing things', required=False)
    parser.add_argument('-f', nargs='+', help='Folds. Default: 0 1 2 3 4', required=False, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    folds = args.f
    folds = [int(i) if i != 'all' else i for i in folds]

    taskname = args.t
    if taskname.startswith("Task"):
        pass
    else:
        try:
            taskid = int(taskname)
        except Exception as e:
            print('-t must be either a Task name (TaskXXX_YYY) or a task id (integer)')
            raise e
        taskname = convert_id_to_task_name(taskid)

    export_pretrained_model(taskname, args.o, args.m, args.tr, args.trc, args.pl, strict=not args.disable_strict,
                            folds=folds)


def export_for_paper():
    output_base = "/media/fabian/DeepLearningData/nnunet_trained_models"
    task_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 24, 27, 29, 35, 48, 55, 61, 38]
    for t in task_ids:
        if t == 61:
            models = ("3d_fullres",)
        else:
            models = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres")
        taskname = convert_id_to_task_name(t)
        print(taskname)
        output_folder = join(output_base, taskname)
        maybe_mkdir_p(output_folder)
        copy_pretrained_models_for_task(taskname, output_folder, models)
        copy_ensembles(taskname, output_folder)
    compress_everything(output_base, 8)
