import os
from multiprocessing import Pool
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer
from nnunetv2.paths import nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.utils import get_caseIDs_from_splitted_dataset_folder, create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.task_name_id_conversion import convert_id_to_task_name


class DatasetFingerprintExtractor(object):
    def __init__(self, task_name_or_id: Union[str, int], num_processes: int = 8):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_by_modality)
        """
        if isinstance(task_name_or_id, int):
            task_name = convert_id_to_task_name(task_name_or_id)
        else:
            task_name = task_name_or_id

        self.input_folder = join(nnUNet_raw, task_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        assert len(images.shape) == 4
        assert len(segmentation.shape) == 4

        rs = np.random.RandomState(seed)

        intensities_per_modality = []
        # we don't use the intensity_statistics_by_modality at all, it's just something that might be nice to have
        intensity_statistics_by_modality = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_modality.append(rs.choice(foreground_pixels, num_samples, replace=True))
            intensity_statistics_by_modality.append({
                'mean': np.mean(foreground_pixels),
                'median': np.median(foreground_pixels),
                'min': np.min(foreground_pixels),
                'max': np.max(foreground_pixels),
                'percentile_99_5': np.percentile(foreground_pixels, 99.5),
                'percentile_00_5': np.percentile(foreground_pixels, 0.5),

            })

        return intensities_per_modality, intensity_statistics_by_modality

    @staticmethod
    def analyze_case(image_files: List[str], segmentation_file: str, reader_writer_class: Type[BaseReaderWriter],
                     num_samples: int = 10000):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

        foreground_intensities_by_modality, foreground_intensity_stats_by_modality = \
            DatasetFingerprintExtractor.collect_foreground_intensities(seg_cropped, data_cropped,
                                                                       num_samples=num_samples)

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return shape_after_crop, spacing, foreground_intensities_by_modality, foreground_intensity_stats_by_modality, \
            relative_size_after_cropping

    def run(self, overwrite_existing: bool = False) -> None:
        properties_file = join(self.input_folder, 'dataset_properties.json')
        if not isfile(properties_file) or overwrite_existing:
            file_suffix = self.dataset_json['file_ending']
            training_identifiers = get_caseIDs_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'),
                                                                            file_suffix)
            reader_writer_class = determine_reader_writer(self.dataset_json,
                                                          join(self.input_folder, 'imagesTr',
                                                               training_identifiers[0] + '_0000' + file_suffix))

            training_images_per_case = create_lists_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'),
                                                                                 file_suffix)
            training_labels_per_case = [join(self.input_folder, 'labelsTr', i + file_suffix) for i in training_identifiers]

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(training_identifiers))

            # DatasetFingerprintExtractor.analyze_case(*list(zip(training_images_per_case, training_labels_per_case,
            #                      [reader_writer_class] * len(training_identifiers),
            #                      [num_foreground_samples_per_case] * len(training_identifiers))
            #                  )[0])

            pool = Pool(self.num_processes)
            results = \
                pool.starmap_async(DatasetFingerprintExtractor.analyze_case,
                             zip(training_images_per_case, training_labels_per_case,
                                 [reader_writer_class] * len(training_identifiers),
                                 [num_foreground_samples_per_case] * len(training_identifiers))
                             )
            results = results.get()
            pool.close()
            pool.join()

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_by_modality = [np.concatenate([r[2][i] for r in results]) for i in range(len(results[0][2]))]
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

            num_modalities = len(self.dataset_json['modality'].keys())
            intensity_statistics_by_modality = {}
            for i in range(num_modalities):
                intensity_statistics_by_modality[i] = {
                    'mean': float(np.mean(foreground_intensities_by_modality[i])),
                    'median': float(np.median(foreground_intensities_by_modality[i])),
                    'std': float(np.std(foreground_intensities_by_modality[i])),
                    'min': float(np.min(foreground_intensities_by_modality[i])),
                    'max': float(np.max(foreground_intensities_by_modality[i])),
                    'percentile_99_5': float(np.percentile(foreground_intensities_by_modality[i], 99.5)),
                    'percentile_00_5': float(np.percentile(foreground_intensities_by_modality[i], 0.5)),
                }

            try:
                save_json({
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    'foreground_intensity_properties_by_modality': intensity_statistics_by_modality,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(2, 10)
    dfe.run(overwrite_existing=True)
