from .abstract_transforms import AbstractTransform, Compose, RndTransform
from .channel_selection_transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, \
    SegChannelMergeTransform, SegChannelRandomSwapTransform, SegChannelRandomDuplicateTransform, \
    SegLabelSelectionBinarizeTransform
from .color_transforms import BrightnessMultiplicativeTransform, BrightnessTransform, ContrastAugmentationTransform, \
    FancyColorTransform, GammaTransform, IlluminationTransform, NormalizeTransform, ClipValueRange, LocalGammaTransform, \
    BrightnessGradientAdditiveTransform
from .crop_and_pad_transforms import CenterCropSegTransform, CenterCropTransform, PadTransform, RandomCropTransform
from .noise_transforms import GaussianBlurTransform, GaussianNoiseTransform, SharpeningTransform, MedianFilterTransform
from .sample_normalization_transforms import CutOffOutliersTransform, RangeTransform, ZeroMeanUnitVarianceTransform

from .utility_transforms import ConvertSegToOnehotTransform, ListToTensor, NumpyToTensor, RenameTransform, \
    ConvertMultiSegToOnehotTransform, ConvertSegToArgmaxTransform, ConvertMultiSegToArgmaxTransform
from .spatial_transforms import ChannelTranslation, MirrorTransform, SpatialTransform, SpatialTransform_2, ZoomTransform, \
    TransposeAxesTransform, ResizeTransform
from .resample_transforms import SimulateLowResolutionTransform

transform_list = [
    AbstractTransform, Compose, RndTransform, DataChannelSelectionTransform,
    SegChannelSelectionTransform, BrightnessMultiplicativeTransform, BrightnessTransform,
    ContrastAugmentationTransform, FancyColorTransform, GammaTransform, IlluminationTransform,
    CenterCropSegTransform, CenterCropTransform, PadTransform, RandomCropTransform,
    GaussianNoiseTransform, GaussianBlurTransform, CutOffOutliersTransform, RangeTransform,
    ZeroMeanUnitVarianceTransform, ChannelTranslation, MirrorTransform, SpatialTransform, ZoomTransform,
    ConvertSegToOnehotTransform, ListToTensor, NumpyToTensor, LocalGammaTransform, BrightnessGradientAdditiveTransform,
    SharpeningTransform, MedianFilterTransform
]
