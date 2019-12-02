import argparse
from nnunet.postprocessing.consolidate_postprocessing import consolidate_folds
from nnunet.utilities.folder_names import get_output_folder_name


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", type=str, required=True, help="U-Net model (2d, 3d_lowres, 3d_fullres or "
                                                               "3d_cascade_fullres)")
    argparser.add_argument("-t", type=str, required=True, help="Task name")
    argparser.add_argument("-tr", type=str, required=False, default=None,
                           help="nnUNetTrainer class. Default: nnUNetTrainerV2, unless 3d_cascade_fullres "
                                "(then it's nnUNetTrainerV2CascadeFullRes)")
    argparser.add_argument("-pl", type=str, required=False, default="nnUNetPlansv2.1",
                           help="Plans name, Default=nnUNetPlansv2.1")
    argparser.add_argument("-val", type=str, required=False, default="validation_raw",
                           help="Validation folder name. Default: validation_raw")

    args = argparser.parse_args()
    model = args.m
    task = args.t
    trainer = args.tr
    plans = args.pl
    val = args.val

    if trainer is None:
        if model == "3d_cascade_fullres":
            trainer = "nnUNetTrainerV2CascadeFullRes"
        else:
            trainer = "nnUNetTrainerV2"

    folder = get_output_folder_name(model, task, trainer, plans, None)

    consolidate_folds(folder, val)
