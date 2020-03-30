from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunet.paths import default_cascade_trainer, default_plans_identifier, default_trainer, network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name


def copy_fold(in_folder: str, out_folder: str):
    shutil.copy(join(in_folder, "debug.json"), join(out_folder, "debug.json"))
    shutil.copy(join(in_folder, "model_final_checkpoint.model"), join(out_folder, "model_final_checkpoint.model"))
    shutil.copy(join(in_folder, "model_final_checkpoint.model.pkl"), join(out_folder, "model_final_checkpoint.model.pkl"))
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


def copy_pretrained_models_for_task(task_name: str, output_directory: str, models: tuple=("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres")):
    nnunet_trainer = default_trainer
    nnunet_trainer_cascade = default_cascade_trainer
    plans_identifier = default_plans_identifier

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


def copy_ensembles(taskname, output_folder, must_have=('nnUNetPlansv2.1', 'nnUNetTrainerV2')):
    ensemble_dir = join(network_training_output_dir, 'ensembles', taskname)
    if not isdir(ensemble_dir):
        print("No ensemble directory found for task", taskname)
        return
    subd = subdirs(ensemble_dir, join=False)
    valid = []
    for s in subd:
        v = True
        for m in must_have:
            if s.find(m) == -1:
                v = False
        if v:
            valid.append(s)
    output_ensemble = join(output_folder, 'ensembles', taskname)
    maybe_mkdir_p(output_ensemble)
    for v in valid:
        this_output = join(output_ensemble, v)
        maybe_mkdir_p(this_output)
        shutil.copy(join(ensemble_dir, v, 'postprocessing.json'), this_output)


if __name__ == "__main__":
    output_base = "/media/fabian/DeepLearningData/nnunet_trained_models"
    task_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 24, 27, 29, 35, 48, 55, 61, 38]
    for t in task_ids:
        if t == 61:
            models = ("3d_fullres", )
        else:
            models = ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres")
        taskname = convert_id_to_task_name(t)
        print(taskname)
        output_folder = join(output_base, taskname)
        maybe_mkdir_p(output_folder)
        copy_pretrained_models_for_task(taskname, output_folder, models)
        copy_ensembles(taskname, output_folder)
