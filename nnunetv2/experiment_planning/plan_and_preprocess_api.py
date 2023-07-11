import shutil
from typing import List, Type, Optional, Tuple, Union

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, load_json

from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name, maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)


def extract_fingerprints(dataset_ids: List[int], fingerprint_extractor_class_name: str = 'DatasetFingerprintExtractor',
                         num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                         clean: bool = True, verbose: bool = True):
    """
    clean = False will not actually run this. This is just a switch for use with nnUNetv2_plan_and_preprocess where
    we don't want to rerun fingerprint extraction every time.
    """
    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              fingerprint_extractor_class_name,
                                                              current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        extract_fingerprint_dataset(d, fingerprint_extractor_class, num_processes, check_dataset_integrity, clean,
                                    verbose)


def plan_experiment_dataset(dataset_id: int,
                            experiment_planner_class: Type[ExperimentPlanner] = ExperimentPlanner,
                            gpu_memory_target_in_gb: float = 8, preprocess_class_name: str = 'DefaultPreprocessor',
                            overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                            overwrite_plans_name: Optional[str] = None) -> dict:
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    kwargs = {}
    if overwrite_plans_name is not None:
        kwargs['plans_name'] = overwrite_plans_name
    return experiment_planner_class(dataset_id,
                                    gpu_memory_target_in_gb=gpu_memory_target_in_gb,
                                    preprocessor_name=preprocess_class_name,
                                    overwrite_target_spacing=[float(i) for i in overwrite_target_spacing] if
                                    overwrite_target_spacing is not None else overwrite_target_spacing,
                                    suppress_transpose=False,  # might expose this later,
                                    **kwargs
                                    ).plan_experiment()


def plan_experiments(dataset_ids: List[int], experiment_planner_class_name: str = 'ExperimentPlanner',
                     gpu_memory_target_in_gb: float = 8, preprocess_class_name: str = 'DefaultPreprocessor',
                     overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                     overwrite_plans_name: Optional[str] = None):
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     experiment_planner_class_name,
                                                     current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        plan_experiment_dataset(d, experiment_planner, gpu_memory_target_in_gb, preprocess_class_name,
                                overwrite_target_spacing, overwrite_plans_name)


def preprocess_dataset(dataset_id: int,
                       plans_identifier: str = 'nnUNetPlans',
                       configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
                       num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
                       verbose: bool = False) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f'Preprocessing dataset {dataset_name}')
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...')
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.")
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, 'gt_segmentations'))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(nnUNet_preprocessed, dataset_name, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)



def preprocess(dataset_ids: List[int],
               plans_identifier: str = 'nnUNetPlans',
               configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
               num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
               verbose: bool = False):
    for d in dataset_ids:
        preprocess_dataset(d, plans_identifier, configurations, num_processes, verbose)
