import math
import os
from abc import ABC, abstractmethod
from typing import List, Union, Type
from typing import Optional, Sequence, Tuple

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle, subfiles

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.utils import unpack_dataset


def comp_blosc2_params(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    bytes_per_pixel: int = 4,
    max_block_nbytes: int = 128 * 1024,
    max_chunk_nbytes: int = 6 * 1024**2,
    max_chunk_to_patch_ratio_per_axis: Optional[float] = 1.5,
    grow_singleton_patch_axes: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Compute Blosc2 block and chunk shapes for 4D arrays with layout:

        image_size = (c, x, y, z)

    patch_size is spatial only:

        patch_size = (x, y, z) for 3D patches
        patch_size = (y, z)    for 2D patches, internally promoted to (1, y, z)

    The channel axis is always kept at 1 for blocks and chunks.

    max_block_nbytes and max_chunk_nbytes are hard caps on the logical useful
    payload, not on any internally padded Blosc2 extent.

    Block and chunk shapes are chosen to roughly follow the image aspect ratio,
    subject to:
    - the patch-size-based per-axis chunk limit
    - the block byte cap
    - the chunk byte cap
    """

    def ceil_pow2(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def prev_pow2_below(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << ((n - 1).bit_length() - 1)

    def nbytes(shape: Sequence[int]) -> int:
        return math.prod(int(i) for i in shape) * int(bytes_per_pixel)

    if len(image_size) != 4:
        raise ValueError(f"image_size must be 4D, i.e. (c, x, y, z). Got {image_size}.")

    if len(patch_size) not in (2, 3):
        raise ValueError(f"patch_size must be 2D or 3D spatial shape. Got {patch_size}.")

    if bytes_per_pixel <= 0:
        raise ValueError("bytes_per_pixel must be positive.")

    if max_block_nbytes <= 0:
        raise ValueError("max_block_nbytes must be positive.")

    if max_chunk_nbytes <= 0:
        raise ValueError("max_chunk_nbytes must be positive.")

    if (
        max_chunk_to_patch_ratio_per_axis is not None
        and max_chunk_to_patch_ratio_per_axis < 1.0
    ):
        raise ValueError("max_chunk_to_patch_ratio_per_axis must be >= 1.0 or None.")

    image_size = tuple(int(i) for i in image_size)

    if any(i <= 0 for i in image_size):
        raise ValueError(f"All image_size entries must be positive. Got {image_size}.")

    is_2d_patch = len(patch_size) == 2

    if is_2d_patch:
        patch_spatial = (1, int(patch_size[0]), int(patch_size[1]))
    else:
        patch_spatial = tuple(int(i) for i in patch_size)

    if any(i <= 0 for i in patch_spatial):
        raise ValueError(f"All patch_size entries must be positive. Got {patch_size}.")

    image_spatial = image_size[1:]

    if max_chunk_to_patch_ratio_per_axis is None:
        target_spatial = image_spatial
    else:
        target_spatial = tuple(
            min(
                image_spatial[ax],
                int(math.floor(max_chunk_to_patch_ratio_per_axis * patch_spatial[ax])),
            )
            for ax in range(3)
        )

    target_spatial = tuple(max(1, int(i)) for i in target_spatial)

    # --------------------
    # Block shape
    # --------------------
    block = [1]
    for p, img in zip(patch_spatial, image_spatial):
        block.append(min(ceil_pow2(p), img))

    effective_block_cap = min(max_block_nbytes, max_chunk_nbytes)

    while nbytes(block) > effective_block_cap:
        candidate_axes = [ax for ax in range(3) if block[ax + 1] > 1]

        if not candidate_axes:
            raise ValueError(
                f"Cannot fit minimum block {tuple(block)} into "
                f"{effective_block_cap} bytes with bytes_per_pixel={bytes_per_pixel}."
            )

        # Shrink the most over-represented axis relative to the target extent.
        # Tie-break toward later axes to preserve the old isotropic default:
        # (1, 64, 64, 32) for 128^3 patches with a 512 KiB block cap.
        picked_axis = max(
            candidate_axes,
            key=lambda ax: (block[ax + 1] / target_spatial[ax], ax),
        )

        block[picked_axis + 1] = max(1, prev_pow2_below(block[picked_axis + 1]))

    # --------------------
    # Chunk shape
    # --------------------
    chunk = block.copy()

    while True:
        candidates = []

        for ax in range(3):
            if is_2d_patch and ax == 0:
                continue

            if patch_spatial[ax] == 1 and not grow_singleton_patch_axes:
                continue

            if chunk[ax + 1] >= image_spatial[ax]:
                continue

            if chunk[ax + 1] >= target_spatial[ax]:
                continue

            candidate = chunk.copy()

            # Grow by one block step, but allow clamping to image boundary
            # or to the target extent.
            candidate[ax + 1] = min(
                candidate[ax + 1] + block[ax + 1],
                image_spatial[ax],
                target_spatial[ax],
            )

            if candidate[ax + 1] == chunk[ax + 1]:
                continue

            candidate_nbytes = nbytes(candidate)

            # Hard cap on logical useful chunk payload.
            if candidate_nbytes > max_chunk_nbytes:
                continue

            # Grow the most under-covered axis relative to the target extent.
            # Tie-break toward the larger original image axis, so for
            # image_spatial=(123, 423, 645), z grows before y when both are tied.
            coverage_ratio = chunk[ax + 1] / target_spatial[ax]

            candidates.append(
                (
                    coverage_ratio,
                    -image_spatial[ax],
                    ax,
                    -candidate_nbytes,
                    candidate,
                )
            )

        if not candidates:
            break

        _, _, _, _, chunk = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )

    return tuple(int(i) for i in block), tuple(int(i) for i in chunk)


class nnUNetBaseDataset(ABC):
    """
    Defines the interface
    """
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__()
        # print('loading dataset')
        if identifiers is None:
            identifiers = self.get_identifiers(folder)
        identifiers.sort()

        self.source_folder = folder
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.identifiers = identifiers

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    @abstractmethod
    def load_case(self, identifier):
        pass

    @staticmethod
    @abstractmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
            ):
        pass

    @staticmethod
    @abstractmethod
    def get_identifiers(folder: str) -> List[str]:
        pass

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True):
        pass


class nnUNetDatasetNumpy(nnUNetBaseDataset):
    def load_case(self, identifier):
        data_npy_file = join(self.source_folder, identifier + '.npy')
        if not isfile(data_npy_file):
            data = np.load(join(self.source_folder, identifier + '.npz'))['data']
        else:
            data = np.load(data_npy_file, mmap_mode='r')

        seg_npy_file = join(self.source_folder, identifier + '_seg.npy')
        if not isfile(seg_npy_file):
            seg = np.load(join(self.source_folder, identifier + '.npz'))['seg']
        else:
            seg = np.load(seg_npy_file, mmap_mode='r')

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_npy_file = join(self.folder_with_segs_from_previous_stage, identifier + '.npy')
            if isfile(prev_seg_npy_file):
                seg_prev = np.load(prev_seg_npy_file, 'r')
            else:
                seg_prev = np.load(join(self.folder_with_segs_from_previous_stage, identifier + '.npz'))['seg']
        else:
            seg_prev = None

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, seg_prev, properties

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + '.npz', seg=seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True):
        return unpack_dataset(folder, True, overwrite_existing, num_processes, verify)


class nnUNetDatasetBlosc2(nnUNetBaseDataset):
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None):
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        blosc2.set_nthreads(1)
        # mmap does not work with Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
        self.mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        dparams = {
            'nthreads': 1
        }
        data_b2nd_file = join(self.source_folder, identifier + '.b2nd')

        data = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, **self.mmap_kwargs)

        seg_b2nd_file = join(self.source_folder, identifier + '_seg.b2nd')
        seg = blosc2.open(urlpath=seg_b2nd_file, mode='r', dparams=dparams, **self.mmap_kwargs)

        if self.folder_with_segs_from_previous_stage is not None:
            prev_seg_b2nd_file = join(self.folder_with_segs_from_previous_stage, identifier + '.b2nd')
            seg_prev = blosc2.open(urlpath=prev_seg_b2nd_file, mode='r', dparams=dparams, **self.mmap_kwargs)
        else:
            seg_prev = None

        properties = load_pickle(join(self.source_folder, identifier + '.pkl'))
        return data, seg, seg_prev, properties

    @staticmethod
    def _select_filter(arr: np.ndarray, blocks, chunks, codec, clevel) -> "blosc2.Filter":
        """
        Pick the better blosc2 filter (NOFILTER vs SHUFFLE) for ``arr`` by trial-compressing
        a representative, centered slab (at most one chunk) both ways and keeping the smaller.

        The best filter depends on whether the data was resampled: continuous/interpolated
        intensities favor SHUFFLE, while quantized (non-resampled) intensities favor NOFILTER.
        This is a per-image property with no robust cheap proxy (the unique-value fraction is
        confounded by image size), so we measure the real objective directly. Ties go to
        NOFILTER; any failure falls back to NOFILTER.
        """
        try:
            shape = tuple(int(s) for s in arr.shape)
            # centered slab of at most one chunk -> cheap, representative (lands on foreground)
            slab_shape = [min(int(c), s) for c, s in zip(chunks, shape)]
            slices = tuple(slice((s - ss) // 2, (s - ss) // 2 + ss) for s, ss in zip(shape, slab_shape))
            slab = np.ascontiguousarray(arr[slices])
            trial_blocks = tuple(max(1, min(int(b), ss)) for b, ss in zip(blocks, slab_shape))

            best_filter, best_bytes = blosc2.Filter.NOFILTER, None
            for f in (blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE):
                cparams = {'codec': codec, 'clevel': clevel, 'nthreads': 4, 'filters': [f]}
                comp = blosc2.asarray(slab, chunks=tuple(slab_shape), blocks=trial_blocks, cparams=cparams)
                cb = comp.schunk.cbytes
                if best_bytes is None or cb < best_bytes:
                    best_bytes, best_filter = cb, f
            return best_filter
        except Exception as e:
            from warnings import warn
            warn(f'_select_filter failed ({e!r}); falling back to NOFILTER.')
            return blosc2.Filter.NOFILTER

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 5,
            codec=blosc2.Codec.LZ4HC,
            filters=None,
            filters_seg=None,
    ):
        if chunks is None or blocks is None:
            from warnings import warn
            blocks, chunks = comp_blosc2_params(data.shape, (128, 128, 128))
            warn(f'Warning: Received empty chunks or blocks. Computed with comp_blosc2_params. This is bad because we '
                 f'do not know the access pattern here (patch size). This should be fixed and not ignored. '
                 f'Raise an issue at github.com/MIC-DKFZ/nnUNet\n'
                 f'data shape: {data.shape}\n'
                 f'chunks {chunks}\n'
                 f'blocks {blocks}\n')

        blosc2.set_nthreads(1)

        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        # Auto-select the filter for data and seg independently (try both, keep the winner)
        # unless a filter pipeline was passed explicitly, in which case all provided filters
        # are applied. Each is selected against its own blocks/chunks.
        if filters is None:
            data_filters = [nnUNetDatasetBlosc2._select_filter(data, blocks, chunks, codec, clevel)]
        else:
            data_filters = list(filters)

        if filters_seg is None:
            seg_filters = [nnUNetDatasetBlosc2._select_filter(seg, blocks_seg, chunks_seg, codec, clevel)]
        else:
            seg_filters = list(filters_seg)

        cparams = {
            'codec': codec,
            'filters': data_filters,
            'nthreads': 4,
            'clevel': clevel,
        }
        cparams_seg = {
            'codec': codec,
            'filters': seg_filters,
            'nthreads': 4,
            'clevel': clevel,
        }

        # print(output_filename_truncated, data.shape, seg.shape, blocks, chunks, blocks_seg, chunks_seg, data.dtype, seg.dtype)

        blosc2.asarray(
            np.ascontiguousarray(data),
            urlpath=output_filename_truncated + '.b2nd',
            chunks=chunks,
            blocks=blocks,
            cparams=cparams,
        )

        blosc2.asarray(
            np.ascontiguousarray(seg),
            urlpath=output_filename_truncated + '_seg.b2nd',
            chunks=chunks_seg,
            blocks=blocks_seg,
            cparams=cparams_seg,
        )

        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def save_seg(
            seg: np.ndarray,
            output_filename_truncated: str,
            chunks_seg=None,
            blocks_seg=None
    ):
        if isfile(output_filename_truncated + '.b2nd'):
            os.remove(output_filename_truncated + '.b2nd')
        blosc2.asarray(seg, urlpath=output_filename_truncated + '.b2nd', chunks=chunks_seg, blocks=blocks_seg)

    @staticmethod
    def get_identifiers(folder: str) -> List[str]:
        """
        returns all identifiers in the preprocessed data folder
        """
        case_identifiers = [i[:-5] for i in os.listdir(folder) if i.endswith(".b2nd") and not i.endswith("_seg.b2nd")]
        return case_identifiers

    @staticmethod
    def unpack_dataset(folder: str, overwrite_existing: bool = False,
                       num_processes: int = default_num_processes,
                       verify: bool = True):
        pass


file_ending_dataset_mapping = {
    'npz': nnUNetDatasetNumpy,
    'b2nd': nnUNetDatasetBlosc2
}


def infer_dataset_class(folder: str) -> Union[Type[nnUNetDatasetBlosc2], Type[nnUNetDatasetNumpy]]:
    file_endings = set([os.path.basename(i).split('.')[-1] for i in subfiles(folder, join=False)])
    if 'pkl' in file_endings:
        file_endings.remove('pkl')
    if 'npy' in file_endings:
        file_endings.remove('npy')
    assert len(file_endings) == 1, (f'Found more than one file ending in the folder {folder}. '
                                    f'Unable to infer nnUNetDataset variant!')
    return file_ending_dataset_mapping[list(file_endings)[0]]
