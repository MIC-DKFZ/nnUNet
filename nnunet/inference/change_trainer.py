from batchgenerators.utilities.file_and_folder_operations import *


def pretend_to_be_nnUNetTrainer(folder, checkpoints=("model_best.model.pkl", "model_final_checkpoint.model.pkl")):
    pretend_to_be_other_trainer(folder, "nnUNetTrainer", checkpoints)


def pretend_to_be_other_trainer(folder, new_trainer_name, checkpoints=("model_best.model.pkl", "model_final_checkpoint.model.pkl")):
    folds = subdirs(folder, prefix="fold_", join=False)
    for c in checkpoints:
        for f in folds:
            checkpoint_file = join(folder, f, c)
            if isfile(checkpoint_file):
                a = load_pickle(checkpoint_file)
                a['name'] = new_trainer_name
                save_pickle(a, checkpoint_file)

