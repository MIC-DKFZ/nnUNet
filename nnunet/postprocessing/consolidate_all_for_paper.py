from nnunet.postprocessing.consolidate_postprocessing import consolidate_folds
from nnunet.utilities.folder_names import get_output_folder_name


def get_datasets():
    configurations_all = {
        "Task01_BrainTumour": ("3d_fullres", "2d"),
        "Task02_Heart": ("3d_fullres", "2d",),
        "Task03_Liver": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task04_Hippocampus": ("3d_fullres", "2d",),
        "Task05_Prostate": ("3d_fullres", "2d",),
        "Task06_Lung": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task07_Pancreas": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task08_HepaticVessel": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task09_Spleen": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task10_Colon": ("3d_cascade_fullres", "3d_fullres", "3d_lowres", "2d"),
        "Task48_KiTS_clean": ("3d_cascade_fullres", "3d_lowres", "3d_fullres", "2d"),
        "Task27_ACDC": ("3d_fullres", "2d",),
        "Task24_Promise": ("3d_fullres", "2d",),
        "Task35_ISBILesionSegmentation": ("3d_fullres", "2d",),
        "Task38_CHAOS_Task_3_5_Variant2": ("3d_fullres", "2d",),
        "Task29_LITS": ("3d_cascade_fullres", "3d_lowres", "2d", "3d_fullres",),
        "Task17_AbdominalOrganSegmentation": ("3d_cascade_fullres", "3d_lowres", "2d", "3d_fullres",),
        "Task55_SegTHOR": ("3d_cascade_fullres", "3d_lowres", "3d_fullres", "2d",),
        "Task56_VerSe": ("3d_cascade_fullres", "3d_lowres", "3d_fullres", "2d",),
    }
    return configurations_all


def get_commands(configurations, regular_trainer="nnUNetTrainerV2", cascade_trainer="nnUNetTrainerV2CascadeFullRes",
                 plans="nnUNetPlansv2.1"):

    node_pool = ["hdf18-gpu%02.0d" % i for i in range(1, 21)] + ["hdf19-gpu%02.0d" % i for i in range(1, 8)] + ["hdf19-gpu%02.0d" % i for i in range(11, 16)]
    ctr = 0
    for task in configurations:
        models = configurations[task]
        for m in models:
            if m == "3d_cascade_fullres":
                trainer = cascade_trainer
            else:
                trainer = regular_trainer

            folder = get_output_folder_name(m, task, trainer, plans, overwrite_training_output_dir="/datasets/datasets_fabian/results/nnUNet")
            node = node_pool[ctr % len(node_pool)]
            print("bsub -m %s -q gputest -L /bin/bash \"source ~/.bashrc && python postprocessing/"
                  "consolidate_postprocessing.py -f" % node, folder, "\"")
            ctr += 1
