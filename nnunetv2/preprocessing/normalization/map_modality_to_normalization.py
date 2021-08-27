from typing import Type

from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization

modality_to_normalization_mapping = {
    'CT': CTNormalization,
    'noNorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization
}


def get_normalization_scheme(modality_name: str) -> Type[ImageNormalization]:
    """
    If we find the modality_name in modality_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = modality_to_normalization_mapping.get(modality_name)
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
