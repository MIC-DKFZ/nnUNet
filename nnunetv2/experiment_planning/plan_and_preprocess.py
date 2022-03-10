import nnunetv2
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.utils import get_preprocessor_class_from_plans
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def extract_fingerprint():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+',
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-np', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run.')
    args, unrecognized_args = parser.parse_known_args()

    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              args.fpe,
                                                              current_module="nnunetv2.experiment_planning")
    for d in args.d:
        d = int(d)

        dataset_name = convert_id_to_dataset_name(d)

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw, dataset_name), args.np)

        fpe = fingerprint_extractor_class(d, args.np)
        fpe.run(overwrite_existing=args.clean)


def plan_experiment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+',
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
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
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    args, unrecognized_args = parser.parse_known_args()
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     args.pl,
                                                     current_module="nnunetv2.experiment_planning")
    for d in args.d:
        d = int(d)
        experiment_planner(d,
                           gpu_memory_target_in_gb=args.gpu_memory_target,
                           preprocessor_name=args.preprocessor_name,
                           plans_name=args.plans_name,
                           overwrite_target_spacing=[float(i) for i in args.overwrite_target_spacing] if
                           args.overwrite_target_spacing is not None else args.overwrite_target_spacing,
                           suppress_transpose=False  # might expose this later
                           ).plan_experiment()


def preprocess():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+',
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset")
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] You can use this to specify a custom plans file that you may have generated')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')
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
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 4 8 (=8 processes for 2d, 4 "
                             "for 3d_fullres and 8 for 3d_lowres if -c is at its default)")
    args, unrecognized_args = parser.parse_known_args()

    np = args.np
    if len(np) == 1:
        np = np * len(args.c)
    if len(np) != len(args.c):
        raise RuntimeError(f'The list provided with -np must either have len 1 or as many elements as there are '
                           f'configurations (see --help). Number of configurations: {len(args.c)}, length of np: '
                           f'{len(np)}')

    for d in args.d:
        d = int(d)
        dataset_name = convert_id_to_dataset_name(d)
        plans_file = join(nnUNet_preprocessed, dataset_name, args.plans_name + '.json')
        plans = load_json(plans_file)
        for n, c in zip(args.np, args.c):
            if c not in plans['configurations'].keys():
                print(
                    f"INFO: Configuration {c} not found in plans file {args.plans_name + '.json'} of dataset {d}. Skipping.")
                continue
            preprocessor = get_preprocessor_class_from_plans(plans, c)()
            preprocessor.run(d, c, args.plans_name, num_processes=n)


def plan_and_preprocess():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+',
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
                             'CHANGE THE DATASET FINGERPRINT EXTRACTOR')
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
    parser.add_argument('-plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
                             '-preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    parser.add_argument('-c', required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')
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
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 4 8 (=8 processes for 2d, 4 "
                             "for 3d_fullres and 8 for 3d_lowres if -c is at its default)")
    args = parser.parse_args()

    # fingerprint extraction
    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              args.fpe,
                                                              current_module="nnunetv2.experiment_planning")
    for d in args.d:
        d = int(d)

        dataset_name = convert_id_to_dataset_name(d)

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw, dataset_name), args.npfp)

        fpe = fingerprint_extractor_class(d, args.npfp)
        fpe.run(overwrite_existing=args.clean)

    # experiment planning
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     args.pl,
                                                     current_module="nnunetv2.experiment_planning")
    for d in args.d:
        d = int(d)
        experiment_planner(d,
                           gpu_memory_target_in_gb=args.gpu_memory_target,
                           preprocessor_name=args.preprocessor_name,
                           plans_name=args.plans_name,
                           overwrite_target_spacing=[float(i) for i in args.overwrite_target_spacing] if
                           args.overwrite_target_spacing is not None else args.overwrite_target_spacing,
                           suppress_transpose=False  # might expose this later
                           ).plan_experiment()

    # preprocessing
    if not args.no_pp:
        np = args.np
        if len(np) == 1:
            np = np * len(args.c)
        if len(np) != len(args.c):
            raise RuntimeError(f'The list provided with -np must either have len 1 or as many elements as there are '
                               f'configurations (see --help). Number of configurations: {len(args.c)}, length of np: '
                               f'{len(np)}')

        for d in args.d:
            d = int(d)
            dataset_name = convert_id_to_dataset_name(d)
            plans_file = join(nnUNet_preprocessed, dataset_name, args.plans_name + '.json')
            plans = load_json(plans_file)
            for n, c in zip(args.np, args.c):
                if c not in plans['configurations'].keys():
                    print(
                        f"INFO: Configuration {c} not found in plans file {args.plans_name + '.json'} of dataset {d}. "
                        f"Skipping.")
                    continue
                preprocessor = get_preprocessor_class_from_plans(plans, c)()
                preprocessor.run(d, c, args.plans_name, num_processes=n)


if __name__ == '__main__':
    plan_and_preprocess()
