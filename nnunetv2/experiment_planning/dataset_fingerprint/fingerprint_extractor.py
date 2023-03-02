import os
from typing import List, Type, Union

import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder


class DatasetFingerprintExtractor(object):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
        """
        images=image with multiple channels = shape (c, x, y(, z))
        """
        assert len(images.shape) == 4
        assert len(segmentation.shape) == 4

        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
        assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
        intensity_statistics_per_channel = []

        # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)
            # sample with replacement so that we don't get issues with cases that have less than num_samples
            # foreground_pixels. We could also just sample less in those cases but that would than cause these
            # training cases to be underrepresented
            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])
            intensity_statistics_per_channel.append({
                'mean': np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                'median': np.median(foreground_pixels) if num_fg > 0 else np.nan,
                'min': np.min(foreground_pixels) if num_fg > 0 else np.nan,
                'max': np.max(foreground_pixels) if num_fg > 0 else np.nan,
                'percentile_99_5': np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan,
                'percentile_00_5': np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan,

            })

        return intensities_per_channel, intensity_statistics_per_channel

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

        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
            DatasetFingerprintExtractor.collect_foreground_intensities(seg_cropped, data_cropped,
                                                                       num_samples=num_samples)

        spacing = properties_images['spacing']

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        return shape_after_crop, spacing, foreground_intensities_per_channel, foreground_intensity_stats_per_channel, \
               relative_size_after_cropping

    def run(self, overwrite_existing: bool = False) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            file_ending = self.dataset_json['file_ending']
            training_identifiers = get_identifiers_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'),
                                                                                file_ending)
            reader_writer_class = determine_reader_writer_from_dataset_json(self.dataset_json,
                                                                            join(self.input_folder, 'imagesTr',
                                                                                 training_identifiers[
                                                                                     0] + '_0000' + file_ending))

            training_images_per_case = create_lists_from_splitted_dataset_folder(join(self.input_folder, 'imagesTr'),
                                                                                 file_ending)
            training_labels_per_case = [join(self.input_folder, 'labelsTr', i + file_ending) for i in
                                        training_identifiers]

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                  len(training_identifiers))

            results = ptqdm(DatasetFingerprintExtractor.analyze_case,
                            (training_images_per_case, training_labels_per_case),
                            processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
                            num_samples=num_foreground_samples_per_case, disable=self.verbose)

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [np.concatenate([r[2][i] for r in results]) for i in
                                                  range(len(results[0][2]))]
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

            num_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                intensity_statistics_per_channel[i] = {
                    'mean': float(np.mean(foreground_intensities_per_channel[i])),
                    'median': float(np.median(foreground_intensities_per_channel[i])),
                    'std': float(np.std(foreground_intensities_per_channel[i])),
                    'min': float(np.min(foreground_intensities_per_channel[i])),
                    'max': float(np.max(foreground_intensities_per_channel[i])),
                    'percentile_99_5': float(np.percentile(foreground_intensities_per_channel[i], 99.5)),
                    'percentile_00_5': float(np.percentile(foreground_intensities_per_channel[i], 0.5)),
                }

            fingerprint = {
                    "spacings": spacings,
                    "shapes_after_crop": shapes_after_crop,
                    'foreground_intensity_properties_per_channel': intensity_statistics_per_channel,
                    "median_relative_size_after_cropping": median_relative_size_after_cropping
                }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(2, 8)
    dfe.run(overwrite_existing=False)
