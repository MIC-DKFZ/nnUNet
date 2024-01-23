from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


def extract_fingerprint_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-np', type=int, default=default_num_processes, required=False,
                        help=f'[OPTIONAL] Number of processes used for fingerprint extraction. '
                             f'Default: {default_num_processes}')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run.')
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints(args.d, args.fpe, args.np, args.verify_dataset_integrity, args.clean, args.verbose)


def plan_experiment_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')
    parser.add_argument('-gpu_memory_target', default=8, type=float, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')
    parser.add_argument('-overwrite_plans_name', default=None, required=False,
                        help='[OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                     args.overwrite_plans_name)


def preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] You can use this to specify a custom plans file that you may have generated')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3d_fullres. Configurations that do not exist for some dataset will be skipped.')
    parser.add_argument('-np', type=int, nargs='+', default=[8, 4, 8], required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args, unrecognized_args = parser.parse_known_args()
    if args.np is None:
        default_np = {
            '2d': 4,
            '3d_lowres': 8,
            '3d_fullres': 4
        }
        np = {default_np[c] if c in default_np.keys() else 4 for c in args.c}
    else:
        np = args.np
    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)


def plan_and_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-npfp', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument('--no_pp', default=False, action='store_true', required=False,
                        help='[OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no '
                             'preprocesing). Useful for debugging.')
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU '
                             'CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!')
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')
    parser.add_argument('-gpu_memory_target', default=8, type=int, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')
    parser.add_argument('-overwrite_plans_name', default=None, required=False,
                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
                             '-preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3d_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3d_fullres. Configurations that do not exist for some dataset will be skipped.')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! '
                             'Recommended for cluster environments')
    args = parser.parse_args()

    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

    # experiment planning
    print('Experiment planning...')
    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name)

    # manage default np
    if args.np is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        np = [default_np[c] if c in default_np.keys() else 4 for c in args.c]
    else:
        np = args.np
    # preprocessing
    if not args.no_pp:
        print('Preprocessing...')
        preprocess(args.d, plans_identifier, args.c, np, args.verbose)


if __name__ == '__main__':
    plan_and_preprocess_entry()
