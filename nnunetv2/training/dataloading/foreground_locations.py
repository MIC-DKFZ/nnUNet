"""
Storage and retrieval of foreground sampling locations.

Historically nnU-Net stored the coordinates used for foreground oversampling inside the per-case
``.pkl`` file as ``properties['class_locations']``: a dict mapping each class (or region) to an
``(N, 4)`` int64 array of voxel coordinates. The dataloader needs *at most one* of those
coordinates per batch item, yet the whole file had to be read and unpickled to get it. On datasets
with many classes this dominates both disk usage and dataloading IO (on TotalSegmentator v2 the
pkl files took 18.26 GB -- 76x more than every segmentation combined -- and a single batch item
cost 6.73 ms to read cold).

This module replaces that with one compressed, partially readable store per configuration folder:

    fg_sampling_locations.b2nd   1D uint64 blosc2 array. Flat (linear) voxel indices, sorted
                                 ascending within each (case, class) run.
    fg_sampling_indptr.npy       int64,  len n_cases + 1. CSR row pointer.
    fg_sampling_class_id.npy     uint16/uint32, len nnz. Index into the class key table.
    fg_sampling_count.npy        uint32, len nnz. Number of coordinates in that run.
    fg_sampling_base.npy         int64,  len n_cases. Where each case's block starts.
    fg_sampling_shape.npy        int64,  (n_cases, 3). Spatial shape, to unravel linear indices.
    fg_sampling_meta.json        Version, case order, class key table, sampling parameters.

Only non-empty (case, class) runs are stored, so the index is genuinely sparse (CSR): it scales
with the number of classes *present per case*, not with the number of classes in the dataset.

Why these parameters (all measured on real TotalSegmentator v2 coordinates):
  * flat linear uint64 -- compresses to exactly the same size as uint32 after the filter pipeline,
    so there is no reason to accept a voxel-count ceiling. A ``(N, 3)`` uint16 layout is 2.8x
    worse and would lose the fast read path below.
  * sorted within each run -- 4x smaller than unsorted. The single biggest lever.
  * ``[SHUFFLE, BYTEDELTA]`` + ZSTD-9 -- 0.72 B/coord, vs 1.26 for SHUFFLE alone and 1.69 for no
    filter.
  * chunks of 8192, blocks of 512 -- the compression ratio is driven by the *block* size, while
    the chunk size only matters through the number of blocks per chunk. This pair sits at the
    cold-read minimum while staying within a few percent of the best achievable size.
  * reads go through ``schunk[i:i + 1]`` rather than ``ndarray[i]`` -- 4-5x faster, because it
    skips the NDArray slicing machinery. Note this is only equivalent to indexing the array
    because the array is 1D.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, load_pickle, save_json

# every file belonging to the store carries this prefix so that the rest of nnU-Net can recognise
# and skip it when scanning a preprocessed folder for cases
FG_SAMPLING_PREFIX = 'fg_sampling_'

LOCATIONS_FILE = FG_SAMPLING_PREFIX + 'locations.b2nd'
INDPTR_FILE = FG_SAMPLING_PREFIX + 'indptr.npy'
CLASS_ID_FILE = FG_SAMPLING_PREFIX + 'class_id.npy'
COUNT_FILE = FG_SAMPLING_PREFIX + 'count.npy'
BASE_FILE = FG_SAMPLING_PREFIX + 'base.npy'
SHAPE_FILE = FG_SAMPLING_PREFIX + 'shape.npy'
META_FILE = FG_SAMPLING_PREFIX + 'meta.json'

STORE_VERSION = 1

# see the module docstring for where these come from
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_BLOCK_SIZE = 512
DEFAULT_CLEVEL = 9

ClassKey = Union[int, Tuple[int, ...]]


def _cparams(clevel: int = DEFAULT_CLEVEL) -> dict:
    return {
        'codec': blosc2.Codec.ZSTD,
        'clevel': clevel,
        'filters': [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
        'nthreads': 1,
    }


def _mmap_kwargs() -> dict:
    # mmap does not work on Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
    return {} if os.name == 'nt' else {'mmap_mode': 'r'}


def _key_to_json(key: ClassKey):
    return list(key) if isinstance(key, (tuple, list)) else int(key)


def _key_from_json(key) -> ClassKey:
    return tuple(int(i) for i in key) if isinstance(key, (list, tuple)) else int(key)


def has_foreground_locations(folder: str) -> bool:
    """The meta file is written last, so its presence means the store is complete."""
    return isfile(join(folder, META_FILE))


def ravel_coords(coords: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """
    (N, 4) or (N, 3) voxel coordinates -> sorted 1D uint64 linear indices into `shape`.

    A leading channel axis (as produced by np.argwhere on a (1, x, y, z) segmentation) is dropped.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError(f'Expected a 2D coordinate array, got shape {coords.shape}')
    if coords.shape[1] == len(shape) + 1:
        coords = coords[:, 1:]
    elif coords.shape[1] != len(shape):
        raise ValueError(f'Coordinate array with {coords.shape[1]} columns does not match shape {tuple(shape)}')
    lin = np.ravel_multi_index([coords[:, i].astype(np.int64) for i in range(len(shape))], tuple(int(i) for i in shape))
    lin.sort()
    return lin.astype(np.uint64, copy=False)


def _unravel(lin: int, shape: Sequence[int]) -> np.ndarray:
    # hot path, so this is done by hand rather than through np.unravel_index
    z = lin % shape[2]
    rest = lin // shape[2]
    y = rest % shape[1]
    x = rest // shape[1]
    return np.array((x, y, z), dtype=np.int64)


class ForegroundLocationsBase:
    """
    Interface used by the dataloader. Two implementations: the blosc2 store below and a fallback
    that reads the legacy per-case pkl files.
    """
    class_keys: List[ClassKey]

    def eligible_classes(self, identifier: str) -> List[ClassKey]:
        """Classes/regions that actually have sampling locations in this case."""
        raise NotImplementedError

    def all_class_keys(self, identifier: str) -> List[ClassKey]:
        """Every class/region key that is valid for this case, whether or not it has locations."""
        raise NotImplementedError

    def count(self, identifier: str, class_key: ClassKey) -> int:
        raise NotImplementedError

    def sample(self, identifier: str, class_key: ClassKey, rng=None) -> np.ndarray:
        """One uniformly drawn voxel coordinate (spatial only, no channel axis) as (3,) int64."""
        raise NotImplementedError


class ForegroundLocations(ForegroundLocationsBase):
    """
    Reader for the blosc2 store. Handles are opened lazily and dropped on pickling, so that
    dataloader worker processes open their own (memory maps and blosc2 handles cannot be shared
    across a fork/spawn boundary).
    """

    def __init__(self, folder: str):
        self.folder = folder
        meta = load_json(join(folder, META_FILE))
        if meta['version'] > STORE_VERSION:
            raise RuntimeError(
                f'The foreground sampling location store in {folder} was written by a newer version of nnU-Net '
                f'(store version {meta["version"]}, this nnU-Net supports up to {STORE_VERSION}). Please update '
                f'nnU-Net or re-run nnUNetv2_extract_sampling_locations.')
        self.version = meta['version']
        self.identifiers: List[str] = list(meta['identifiers'])
        self.class_keys: List[ClassKey] = [_key_from_json(k) for k in meta['class_keys']]
        self.sampling_parameters: dict = meta.get('sampling_parameters', {})
        self._key_to_id = {k: i for i, k in enumerate(self.class_keys)}
        self._case_to_idx = {c: i for i, c in enumerate(self.identifiers)}
        self._handles = None

    # ------------------------------------------------------------------ handles

    def _ensure_open(self):
        if self._handles is not None:
            return
        blosc2.set_nthreads(1)
        arr = blosc2.open(urlpath=join(self.folder, LOCATIONS_FILE), mode='r',
                          dparams={'nthreads': 1}, **_mmap_kwargs())
        self._handles = {
            'schunk': arr.schunk,
            'array': arr,  # keep a reference alive; the schunk does not own the file
            'indptr': np.load(join(self.folder, INDPTR_FILE), mmap_mode='r'),
            'class_id': np.load(join(self.folder, CLASS_ID_FILE), mmap_mode='r'),
            'count': np.load(join(self.folder, COUNT_FILE), mmap_mode='r'),
            'base': np.load(join(self.folder, BASE_FILE), mmap_mode='r'),
            'shape': np.load(join(self.folder, SHAPE_FILE), mmap_mode='r'),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handles = None

    # ------------------------------------------------------------------ lookups

    def _row(self, identifier: str) -> Tuple[int, int, int]:
        self._ensure_open()
        try:
            c = self._case_to_idx[identifier]
        except KeyError:
            raise KeyError(f'Case {identifier} is not part of the foreground sampling location store in '
                           f'{self.folder}. It was probably added after the store was built. Re-run '
                           f'nnUNetv2_extract_sampling_locations for this dataset.')
        h = self._handles
        return c, int(h['indptr'][c]), int(h['indptr'][c + 1])

    def case_shape(self, identifier: str) -> np.ndarray:
        c, _, _ = self._row(identifier)
        return np.asarray(self._handles['shape'][c])

    def eligible_classes(self, identifier: str) -> List[ClassKey]:
        _, s, e = self._row(identifier)
        return [self.class_keys[i] for i in self._handles['class_id'][s:e]]

    def all_class_keys(self, identifier: str) -> List[ClassKey]:
        return list(self.class_keys)

    def _locate(self, identifier: str, class_key: ClassKey) -> Optional[Tuple[int, int]]:
        """-> (offset into the coordinate array, number of coordinates), or None if absent."""
        cid = self._key_to_id.get(class_key)
        if cid is None:
            # callers may hand us keys built from numpy scalars (LabelManager does). Those usually hash like
            # plain ints, but normalise explicitly rather than relying on that.
            cid = self._key_to_id.get(_key_from_json(class_key)) if isinstance(class_key, (tuple, list, np.integer)) \
                else None
            if cid is None:
                return None
        c, s, e = self._row(identifier)
        h = self._handles
        row = h['class_id'][s:e]
        j = int(np.searchsorted(row, cid))
        if j >= len(row) or int(row[j]) != cid:
            return None
        # offsets are not stored; they are recovered by summing the counts that precede this run
        # inside the case. That is a handful of additions and saves 8 bytes per stored run.
        offset = int(h['base'][c]) + int(h['count'][s:s + j].sum())
        return offset, int(h['count'][s + j])

    def count(self, identifier: str, class_key: ClassKey) -> int:
        loc = self._locate(identifier, class_key)
        return 0 if loc is None else loc[1]

    def sample(self, identifier: str, class_key: ClassKey, rng=None) -> np.ndarray:
        loc = self._locate(identifier, class_key)
        if loc is None:
            raise KeyError(f'Case {identifier} has no sampling locations for class {class_key}')
        offset, n = loc
        rng = np.random if rng is None else rng
        i = offset + int(rng.randint(n) if hasattr(rng, 'randint') else rng.integers(n))
        lin = int(np.frombuffer(self._handles['schunk'][i:i + 1], dtype=np.uint64)[0])
        return _unravel(lin, self._handles['shape'][self._case_to_idx[identifier]])

    def all_locations(self, identifier: str, class_key: ClassKey) -> np.ndarray:
        """
        All coordinates of one (case, class) as an (N, 3) int64 array. Not used during training --
        this is for inspection, debugging and testing.
        """
        loc = self._locate(identifier, class_key)
        if loc is None:
            return np.zeros((0, 3), dtype=np.int64)
        offset, n = loc
        lin = np.frombuffer(self._handles['schunk'][offset:offset + n], dtype=np.uint64).astype(np.int64)
        shape = self._handles['shape'][self._case_to_idx[identifier]]
        return np.stack(np.unravel_index(lin, tuple(int(i) for i in shape)), axis=1)


class LegacyForegroundLocations(ForegroundLocationsBase):
    """
    Backwards compatible fallback for datasets preprocessed before the store existed. Reads
    ``properties['class_locations']`` out of the per-case pkl, exactly as nnU-Net used to.

    A single-entry cache keeps the cost at one pkl read per case per batch item (which is what the
    old dataloader paid), rather than one per interface call.
    """

    def __init__(self, folder: str):
        self.folder = folder
        self._cached_identifier = None
        self._cached = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cached_identifier'] = None
        state['_cached'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get(self, identifier: str) -> Dict[ClassKey, np.ndarray]:
        if identifier != self._cached_identifier:
            props = load_pickle(join(self.folder, identifier + '.pkl'))
            if 'class_locations' not in props:
                raise RuntimeError(
                    f'{identifier}.pkl in {self.folder} contains neither a foreground sampling location store nor '
                    f'the legacy class_locations entry. Run nnUNetv2_extract_sampling_locations for this dataset.')
            self._cached = props['class_locations']
            self._cached_identifier = identifier
        return self._cached

    @property
    def class_keys(self) -> List[ClassKey]:
        if self._cached is None:
            raise RuntimeError('class_keys is only available after a case has been accessed')
        return list(self._cached.keys())

    def eligible_classes(self, identifier: str) -> List[ClassKey]:
        cl = self._get(identifier)
        return [k for k in cl.keys() if len(cl[k]) > 0]

    def all_class_keys(self, identifier: str) -> List[ClassKey]:
        return list(self._get(identifier).keys())

    def count(self, identifier: str, class_key: ClassKey) -> int:
        cl = self._get(identifier)
        return len(cl[class_key]) if class_key in cl else 0

    def sample(self, identifier: str, class_key: ClassKey, rng=None) -> np.ndarray:
        cl = self._get(identifier)
        voxels = cl[class_key]
        rng = np.random if rng is None else rng
        i = int(rng.randint(len(voxels)) if hasattr(rng, 'randint') else rng.integers(len(voxels)))
        # legacy coordinates carry a leading channel axis
        return np.asarray(voxels[i][1:], dtype=np.int64)

    def all_locations(self, identifier: str, class_key: ClassKey) -> np.ndarray:
        cl = self._get(identifier)
        if class_key not in cl or len(cl[class_key]) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(cl[class_key], dtype=np.int64)[:, 1:]


_WARNED_FOLDERS = set()


def get_foreground_locations(folder: str, verbose: bool = True) -> ForegroundLocationsBase:
    """
    Open the foreground sampling location store for a preprocessed configuration folder, falling
    back to the legacy per-case pkl files when no store is present.
    """
    if has_foreground_locations(folder):
        return ForegroundLocations(folder)
    if verbose and folder not in _WARNED_FOLDERS:
        _WARNED_FOLDERS.add(folder)
        print(
            f'\n#######################################################################\n'
            f'INFO: {folder} has no foreground sampling location store, falling back to reading the legacy\n'
            f'class_locations entries from the per-case pkl files. Training works, but dataloading is slower and\n'
            f'these files take up a lot of disk space (they can be tens of GB on datasets with many classes).\n'
            f'You can migrate this dataset in place, without re-running preprocessing, with:\n'
            f'    nnUNetv2_extract_sampling_locations -d DATASET_ID\n'
            f'#######################################################################\n')
    return LegacyForegroundLocations(folder)


class ForegroundLocationsWriter:
    """
    Builds the store incrementally so that a multiprocessed extraction pass can stream results to
    disk as they arrive instead of holding everything in RAM.

    Cases may be added in any order -- the index records an explicit offset per (case, class), so
    only the runs themselves need to stay contiguous. Coordinates are buffered and flushed in whole
    multiples of the chunk size; flushing unaligned would force blosc2 to rewrite the partial tail
    chunk on every call (measured 1.4x slower over a full dataset for no benefit).
    """

    def __init__(self, folder: str, class_keys: Sequence[ClassKey],
                 chunk_size: int = DEFAULT_CHUNK_SIZE, block_size: int = DEFAULT_BLOCK_SIZE,
                 clevel: int = DEFAULT_CLEVEL, sampling_parameters: Optional[dict] = None,
                 flush_every_n_chunks: int = 32):
        self.folder = folder
        self.class_keys = [_key_from_json(_key_to_json(k)) for k in class_keys]
        self._key_to_id = {k: i for i, k in enumerate(self.class_keys)}
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.sampling_parameters = sampling_parameters if sampling_parameters is not None else {}
        self._flush_threshold = chunk_size * flush_every_n_chunks

        self._class_id_dtype = np.uint16 if len(self.class_keys) <= np.iinfo(np.uint16).max else np.uint32

        for f in (LOCATIONS_FILE, INDPTR_FILE, CLASS_ID_FILE, COUNT_FILE, BASE_FILE, SHAPE_FILE, META_FILE):
            p = join(folder, f)
            if isfile(p):
                os.remove(p)

        blosc2.set_nthreads(1)
        self._array = blosc2.zeros(shape=(0,), dtype=np.uint64, chunks=(chunk_size,), blocks=(block_size,),
                                   urlpath=join(folder, LOCATIONS_FILE), cparams=_cparams(clevel))
        self._written = 0          # coordinates committed to the blosc2 array
        self._total = 0            # coordinates accepted (committed + buffered)
        self._buffer: List[np.ndarray] = []
        self._buffered = 0

        self._identifiers: List[str] = []
        self._shapes: List[Tuple[int, int, int]] = []
        self._bases: List[int] = []
        self._row_class_ids: List[List[int]] = []
        self._row_counts: List[List[int]] = []
        self._finalized = False

    def add(self, identifier: str, shape: Sequence[int], locations: Dict[ClassKey, np.ndarray]):
        """
        `locations` maps class key -> 1D uint64 array of sorted linear indices (see `ravel_coords`).
        Empty entries are dropped; the CSR index only stores runs that actually have coordinates.
        """
        if self._finalized:
            raise RuntimeError('This writer has already been finalized')
        if len(shape) != 3:
            raise ValueError(f'Expected a 3D spatial shape, got {tuple(shape)}')

        self._identifiers.append(identifier)
        self._shapes.append(tuple(int(i) for i in shape))
        self._bases.append(self._total)

        class_ids, counts = [], []
        # sorted by class id so that lookups can binary-search the row
        for key in sorted((k for k, v in locations.items() if len(v) > 0), key=lambda k: self._key_to_id[k]):
            v = np.ascontiguousarray(locations[key], dtype=np.uint64)
            class_ids.append(self._key_to_id[key])
            counts.append(len(v))
            self._buffer.append(v)
            self._buffered += len(v)
            self._total += len(v)
        self._row_class_ids.append(class_ids)
        self._row_counts.append(counts)

        if self._buffered >= self._flush_threshold:
            self._flush(aligned_only=True)

    def _flush(self, aligned_only: bool):
        if self._buffered == 0:
            return
        data = np.concatenate(self._buffer) if len(self._buffer) > 1 else self._buffer[0]
        n = (len(data) // self.chunk_size) * self.chunk_size if aligned_only else len(data)
        if n == 0:
            self._buffer = [data]
            return
        self._array.resize((self._written + n,))
        self._array[self._written:self._written + n] = data[:n]
        self._written += n
        rest = data[n:]
        self._buffer = [rest] if len(rest) else []
        self._buffered = len(rest)

    def finalize(self):
        if self._finalized:
            return
        self._flush(aligned_only=False)
        assert self._written == self._total, (self._written, self._total)

        n_cases = len(self._identifiers)
        indptr = np.zeros(n_cases + 1, dtype=np.int64)
        np.cumsum([len(r) for r in self._row_class_ids], out=indptr[1:])
        class_id = np.array([c for row in self._row_class_ids for c in row], dtype=self._class_id_dtype)
        count = np.array([c for row in self._row_counts for c in row], dtype=np.uint32)
        base = np.array(self._bases, dtype=np.int64)
        shape = np.array(self._shapes, dtype=np.int64).reshape(n_cases, 3)

        np.save(join(self.folder, INDPTR_FILE), indptr)
        np.save(join(self.folder, CLASS_ID_FILE), class_id)
        np.save(join(self.folder, COUNT_FILE), count)
        np.save(join(self.folder, BASE_FILE), base)
        np.save(join(self.folder, SHAPE_FILE), shape)
        # written last: its presence marks the store as complete
        save_json({
            'version': STORE_VERSION,
            'identifiers': self._identifiers,
            'class_keys': [_key_to_json(k) for k in self.class_keys],
            'num_coordinates': int(self._total),
            'chunk_size': self.chunk_size,
            'block_size': self.block_size,
            'sampling_parameters': self.sampling_parameters,
        }, join(self.folder, META_FILE), sort_keys=False)
        self._finalized = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
