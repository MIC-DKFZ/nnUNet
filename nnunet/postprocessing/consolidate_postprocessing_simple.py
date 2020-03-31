import argparse
from nnunet.postprocessing.consolidate_postprocessing import consolidate_folds
from nnunet.utilities.folder_names import get_output_folder_name
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.paths import default_cascade_trainer, default_trainer, default_plans_identifier


def main():
    argparser = argparse.ArgumentParser(usage="Used to determine the postprocessing for a trained model. Useful for "
                                              "when the best configuration (2d, 3d_fullres etc) as selected manually.")
    argparser.add_argument("-m", type=str, required=True, help="U-Net model (2d, 3d_lowres, 3d_fullres or "
                                                               "3d_cascade_fullres)")
    argparser.add_argument("-t", type=str, required=True, help="Task name or id")
    argparser.add_argument("-tr", type=str, required=False, default=None,
                           help="nnUNetTrainer class. Default: %s, unless 3d_cascade_fullres "
                                "(then it's %s)" % (default_trainer, default_cascade_trainer))
    argparser.add_argument("-pl", type=str, required=False, default=default_plans_identifier,
                           help="Plans name, Default=%s" % default_plans_identifier)
    argparser.add_argument("-val", type=str, required=False, default="validation_raw",
                           help="Validation folder name. Default: validation_raw")

    args = argparser.parse_args()
    model = args.m
    task = args.t
    trainer = args.tr
    plans = args.pl
    val = args.val

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if trainer is None:
        if model == "3d_cascade_fullres":
            trainer = "nnUNetTrainerV2CascadeFullRes"
        else:
            trainer = "nnUNetTrainerV2"

    folder = get_output_folder_name(model, task, trainer, plans, None)

    consolidate_folds(folder, val)


if __name__ == "__main__":
    main()
