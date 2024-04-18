from typing import Type

from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization

channel_name_to_normalization_mapping = {
    'ct': CTNormalization,
    'nonorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_to_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name.casefold())
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
