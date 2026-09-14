"""
Storage and retrieval of foreground sampling locations.

Report: /home/isensee/git_repos/random_projects/projects/2026-09_preprocessing_io_speedups/REPORT.md
Covers this store and the resampling fix on the same branch. Records why preallocating and
memory-mapped writing made the store *slower* on NFS, what the parity runs still have to show
(the store is distributionally but not bit-identical), and the archive location.

Historically nnU-Net stored the coordinates used for foreground oversampling inside the per-case
``.pkl`` file as ``properties['class_locations']``: a dict mapping each class (or region) to an
``(N, 4)`` int64 array of voxel coordinates. The dataloader needs *at most one* of those
coordinates per batch item, yet the whole file had to be read and unpickled to get it. On datasets
with many classes this dominates both disk usage and dataloading IO (on TotalSegmentator v2 the
pkl files took 18.26 GB -- 76x more than every segmentation combined -- and a single batch item
cost 6.73 ms to read cold).

This module replaces that with one compressed, partially readable store per configuration folder.
It lives in its own subdirectory so that nothing scanning that folder for cases has to know about
it (directories are invisible both to ``subfiles`` and to the ``os.listdir`` + suffix scans in
nnunet_dataset.py):

    fg_sampling/locations.b2nd   1D uint64 blosc2 array. Flat (linear) voxel indices, sorted
                                 ascending within each (case, class) run.
    fg_sampling/indptr.npy       int64,  len n_cases + 1. CSR row pointer.
    fg_sampling/class_id.npy     uint16/uint32, len nnz. Index into the class key table.
    fg_sampling/count.npy        uint32, len nnz. Number of coordinates in that run.
    fg_sampling/base.npy         int64,  len n_cases. Where each case's block starts.
    fg_sampling/shape.npy        int64,  (n_cases, 3). Spatial shape, to unravel linear indices.
    fg_sampling/meta.json        Version, case order, class key table, sampling parameters.

Only non-empty (case, class) runs are stored, so the index is genuinely sparse (CSR): it scales
with the number of classes *present per case*, not with the number of classes in the dataset.

The blosc2 parameters below were tuned on real TotalSegmentator v2 coordinates;
documentation/reference/preprocessed-data-format.md carries the measurements. The two decisions
that are not visible in the constants: coordinates are sorted within each run (4x smaller than
unsorted, by far the biggest lever), and reads go through ``schunk[i:i + 1]`` rather than
``ndarray[i]`` (4-5x faster because it skips the NDArray slicing machinery; equivalent only
because the array is 1D).
"""
import multiprocessing
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple, Union

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json, load_pickle, \
    maybe_mkdir_p, save_json

# the store lives in this subdirectory of the preprocessed configuration folder
FG_SAMPLING_DIRNAME = 'fg_sampling'

LOCATIONS_FILE = 'locations.b2nd'
INDPTR_FILE = 'indptr.npy'
CLASS_ID_FILE = 'class_id.npy'
COUNT_FILE = 'count.npy'
BASE_FILE = 'base.npy'
SHAPE_FILE = 'shape.npy'
META_FILE = 'meta.json'

STORE_VERSION = 1

# see the module docstring for where these come from
DEFAULT_CHUNK_SIZE = 32768
DEFAULT_BLOCK_SIZE = 512
# Above this compressed size the store is no longer built in RAM (see ForegroundLocationsWriter).
# NOTE: to_cframe() transiently holds a second copy, so the peak is about twice this. 32 GiB is
# therefore a ~64 GiB peak, chosen so that the slow on-disk fallback is effectively unreachable -
# a 32 GiB store is ~40 billion coordinates, roughly 70x TotalSegmentator v2.
DEFAULT_MAX_IN_MEMORY_BYTES = 32 * 1024 ** 3
DEFAULT_CLEVEL = 9
# writing is a single-process funnel at the end of a multiprocessed extraction pass, so unlike the
# read path it should use several threads (measured 3.6 -> 16.6 Mcoord/s going from 1 to 8, with
# byte-identical output)
DEFAULT_WRITE_NTHREADS = min(os.cpu_count() or 1, 8)

# mmap does not work on Windows -> https://github.com/MIC-DKFZ/nnUNet/issues/2723
MMAP_KWARGS = {} if os.name == 'nt' else {'mmap_mode': 'r'}

ClassKey = Union[int, Tuple[int, ...]]


def store_folder(configuration_folder: str) -> str:
    """Where the store of a preprocessed configuration folder lives."""
    return join(configuration_folder, FG_SAMPLING_DIRNAME)


def _cparams(clevel: int = DEFAULT_CLEVEL, nthreads: int = 1) -> dict:
    return {
        'codec': blosc2.Codec.ZSTD,
        'clevel': clevel,
        'filters': [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
        'nthreads': nthreads,
    }


def _key_to_json(key: ClassKey):
    return list(key) if isinstance(key, (tuple, list)) else int(key)


def normalize_class_key(key) -> ClassKey:
    """
    Canonical, hashable form of a class/region key: a plain int, or a tuple of plain ints. Regions
    arrive as lists from dataset.json and as numpy scalars from the LabelManager; the store keys
    everything by this form so that producer and consumer cannot disagree.
    """
    return tuple(int(i) for i in key) if isinstance(key, (tuple, list)) else int(key)


def _draw_index(rng, n: int) -> int:
    """One index in [0, n). Accepts np.random / RandomState (randint) as well as Generator (integers)."""
    if rng is None:
        rng = np.random
    return int(rng.randint(n) if hasattr(rng, 'randint') else rng.integers(n))


def has_foreground_locations(folder: str) -> bool:
    """The meta file is written last, so its presence means the store is complete."""
    return isfile(join(store_folder(folder), META_FILE))


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


def _unravel(lin: int, shape: Tuple[int, int, int]) -> np.ndarray:
    # hot path, so this is done by hand rather than through np.unravel_index (measured 0.55 vs
    # 1.40 us against a plain int tuple; `shape` is cached as one to keep it that way)
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
        self.configuration_folder = folder
        self.folder = store_folder(folder)
        meta = load_json(join(self.folder, META_FILE))
        if meta['version'] > STORE_VERSION:
            raise RuntimeError(
                f'The foreground sampling location store in {self.folder} was written by a newer version of nnU-Net '
                f'(store version {meta["version"]}, this nnU-Net supports up to {STORE_VERSION}). Please update '
                f'nnU-Net or re-run nnUNetv2_extract_sampling_locations.')
        self.version = meta['version']
        self.identifiers: List[str] = list(meta['identifiers'])
        self.class_keys: List[ClassKey] = [normalize_class_key(k) for k in meta['class_keys']]
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
                          dparams={'nthreads': 1}, **MMAP_KWARGS)
        # the index arrays are small (< 1 MB even for TotalSegmentator v2) and every lookup touches
        # them, so they are read into RAM rather than memory mapped: page faults on the hot path
        # cost more than the memory they would save
        count = np.load(join(self.folder, COUNT_FILE))
        # run offsets are not stored. Recovering them by summing the counts that precede a run is
        # O(classes present in the case) per lookup, so the prefix sum is built once here instead
        # (measured 2.26 -> 0.11 us per lookup on a 100+ class dataset).
        cumulative_count = np.zeros(len(count) + 1, dtype=np.int64)
        cumulative_count[1:] = np.cumsum(count, dtype=np.int64)
        self._handles = {
            'schunk': arr.schunk,
            'array': arr,  # keep a reference alive; the schunk does not own the file
            'indptr': np.load(join(self.folder, INDPTR_FILE)),
            'class_id': np.load(join(self.folder, CLASS_ID_FILE)),
            'count': count,
            'cumulative_count': cumulative_count,
            'base': np.load(join(self.folder, BASE_FILE)),
            # plain int tuples; see _unravel
            'shapes': [tuple(int(i) for i in s) for s in np.load(join(self.folder, SHAPE_FILE))],
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

    def case_shape(self, identifier: str) -> Tuple[int, int, int]:
        c, _, _ = self._row(identifier)
        return self._handles['shapes'][c]

    def eligible_classes(self, identifier: str) -> List[ClassKey]:
        _, s, e = self._row(identifier)
        # .tolist() converts the row in one C-level pass; iterating the array instead boxes one
        # numpy scalar per class (measured 10.9 -> 1.3 us at 104 classes)
        return [self.class_keys[i] for i in self._handles['class_id'][s:e].tolist()]

    def _locate(self, identifier: str, class_key: ClassKey) -> Optional[Tuple[int, int, int]]:
        """-> (case index, offset into the coordinate array, number of coordinates), or None if absent."""
        cid = self._key_to_id.get(class_key)
        if cid is None:
            return None
        c, s, e = self._row(identifier)
        h = self._handles
        row = h['class_id'][s:e]
        j = int(np.searchsorted(row, cid))
        if j >= len(row) or int(row[j]) != cid:
            return None
        cumulative_count = h['cumulative_count']
        offset = int(h['base'][c]) + int(cumulative_count[s + j] - cumulative_count[s])
        return c, offset, int(h['count'][s + j])

    def count(self, identifier: str, class_key: ClassKey) -> int:
        loc = self._locate(identifier, class_key)
        return 0 if loc is None else loc[2]

    def sample(self, identifier: str, class_key: ClassKey, rng=None) -> np.ndarray:
        loc = self._locate(identifier, class_key)
        if loc is None:
            raise KeyError(f'Case {identifier} has no sampling locations for class {class_key}')
        c, offset, n = loc
        i = offset + _draw_index(rng, n)
        lin = int(np.frombuffer(self._handles['schunk'][i:i + 1], dtype=np.uint64)[0])
        return _unravel(lin, self._handles['shapes'][c])

    def all_locations(self, identifier: str, class_key: ClassKey) -> np.ndarray:
        """
        All coordinates of one (case, class) as an (N, 3) int64 array. Not used during training --
        this is for inspection, debugging and testing.
        """
        loc = self._locate(identifier, class_key)
        if loc is None:
            return np.zeros((0, 3), dtype=np.int64)
        c, offset, n = loc
        lin = np.frombuffer(self._handles['schunk'][offset:offset + n], dtype=np.uint64).astype(np.int64)
        return np.stack(np.unravel_index(lin, self._handles['shapes'][c]), axis=1)


class LegacyForegroundLocations(ForegroundLocationsBase):
    """
    Backwards compatible fallback for datasets preprocessed before the store existed. Reads
    ``properties['class_locations']`` out of the per-case pkl, exactly as nnU-Net used to.

    A single-entry cache keeps the cost at one pkl read per case per batch item (which is what the
    old dataloader paid), rather than one per interface call. Note that this means one case's
    coordinates stay resident per dataloader worker; on legacy datasets with many classes that is
    tens of MB per worker, which is one more reason to migrate.
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

    def count(self, identifier: str, class_key: ClassKey) -> int:
        cl = self._get(identifier)
        return len(cl[class_key]) if class_key in cl else 0

    def sample(self, identifier: str, class_key: ClassKey, rng=None) -> np.ndarray:
        voxels = self._get(identifier)[class_key]
        # legacy coordinates carry a leading channel axis
        return np.asarray(voxels[_draw_index(rng, len(voxels))][1:], dtype=np.int64)

    def all_locations(self, identifier: str, class_key: ClassKey) -> np.ndarray:
        cl = self._get(identifier)
        if class_key not in cl or len(cl[class_key]) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(cl[class_key], dtype=np.int64)[:, 1:]


def get_foreground_locations(folder: str) -> ForegroundLocationsBase:
    """
    Open the foreground sampling location store for a preprocessed configuration folder, falling
    back to the legacy per-case pkl files when no store is present.

    Silent by design: this is resolved lazily, which means it first runs inside the dataloader
    worker processes. Announcing a missing store from here would print once per worker. See
    announce_missing_store(), which the dataset calls from the process that constructs it.
    """
    if has_foreground_locations(folder):
        return ForegroundLocations(folder)
    return LegacyForegroundLocations(folder)


_ANNOUNCED_FOLDERS = set()


def announce_missing_store(folder: str) -> bool:
    """
    Tell the user once that `folder` has no sampling location store and how to migrate it.
    Returns whether anything was printed (that is what the tests assert on).

    Call this from the process that *constructs* the dataset. Dataloader workers inherit a copy of
    this module's globals, so a module-level set alone does not make the message unique -- every
    worker would print it again. We therefore stay quiet in child processes and in non-zero DDP
    ranks, and rely on the main process having announced it.
    """
    if multiprocessing.parent_process() is not None:      # a dataloader/export worker
        return False
    if os.environ.get('LOCAL_RANK', '0') != '0':          # a secondary DDP rank
        return False
    if folder in _ANNOUNCED_FOLDERS:
        return False
    # mark before the isfile check so repeated constructions of the same folder cost nothing
    _ANNOUNCED_FOLDERS.add(folder)
    if has_foreground_locations(folder):
        return False
    print(
        f'\n#######################################################################\n'
        f'INFO: {folder} has no foreground sampling location store, falling back to reading the legacy\n'
        f'class_locations entries from the per-case pkl files. Training works, but dataloading is slower and\n'
        f'these files take up a lot of disk space (they can be tens of GB on datasets with many classes).\n'
        f'You can migrate this dataset in place, without re-running preprocessing, with:\n'
        f'    nnUNetv2_extract_sampling_locations -d DATASET_ID\n'
        f'#######################################################################\n')
    return True


class ForegroundLocationsWriter:
    """
    Builds the store incrementally so that a multiprocessed extraction pass can stream results to
    disk as they arrive instead of holding everything in RAM.

    Cases may be added in any order; the CSR index records each case's block offset explicitly, so
    only the runs within one case need to stay contiguous. Coordinates are buffered and flushed in
    whole multiples of the chunk size; flushing unaligned would force blosc2 to rewrite the partial
    tail chunk on every call (measured 1.4x slower over a full dataset for no benefit).

    The array is built **in memory** and serialised to disk once, in finalize(). Writing it
    incrementally to a urlpath-backed array is fine locally but pathological on a network
    filesystem: blosc2 pays a fixed per-chunk cost there (~16 ms/chunk measured on NFS, against
    ~0.09 ms locally) because appending a chunk rewrites the frame's offset trailer. On NFS,
    writing 7.8 MB took 14.8 s incrementally versus 1.1 s built in RAM and written with a single
    to_cframe() + write() - 13x, and the gap grows with dataset size. Neither preallocating the
    array nor memory-mapped writing nor larger flushes fixed it; only removing the per-chunk
    filesystem traffic did.

    Datasets whose store would not fit in RAM spill to the incremental on-disk path
    (max_in_memory_bytes). That is slow on a network filesystem, but it is bounded memory.
    """

    def __init__(self, folder: str, class_keys: Sequence[ClassKey],
                 chunk_size: int = DEFAULT_CHUNK_SIZE, block_size: int = DEFAULT_BLOCK_SIZE,
                 clevel: int = DEFAULT_CLEVEL, sampling_parameters: Optional[dict] = None,
                 flush_every_n_chunks: int = 32, nthreads: int = DEFAULT_WRITE_NTHREADS,
                 max_in_memory_bytes: int = DEFAULT_MAX_IN_MEMORY_BYTES):
        self.folder = store_folder(folder)
        self.class_keys = [normalize_class_key(k) for k in class_keys]
        self._key_to_id = {k: i for i, k in enumerate(self.class_keys)}
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.sampling_parameters = sampling_parameters if sampling_parameters is not None else {}
        self._flush_threshold = chunk_size * flush_every_n_chunks

        self._class_id_dtype = np.uint16 if len(self.class_keys) <= np.iinfo(np.uint16).max else np.uint32

        # start from an empty directory. This also clears leftovers of a store written by a
        # different version, whose file names we may not know
        if isdir(self.folder):
            shutil.rmtree(self.folder)
        maybe_mkdir_p(self.folder)

        self._clevel = clevel
        self._nthreads = nthreads
        self._max_in_memory_bytes = max_in_memory_bytes
        self._locations_path = join(self.folder, LOCATIONS_FILE)
        self._on_disk = False

        blosc2.set_nthreads(nthreads)
        # no urlpath: this lives in RAM until finalize(). See the class docstring.
        self._array = blosc2.zeros(shape=(0,), dtype=np.uint64, chunks=(chunk_size,), blocks=(block_size,),
                                   cparams=_cparams(clevel, nthreads))
        self._written = 0          # coordinates committed to the blosc2 array
        self._total = 0            # coordinates accepted (committed + buffered)
        self._buffer: List[np.ndarray] = []

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
            self._total += len(v)
        self._row_class_ids.append(class_ids)
        self._row_counts.append(counts)

        if self._total - self._written >= self._flush_threshold:
            self._flush(aligned_only=True)

    def _spill_to_disk(self):
        """
        The store got too big to hold in RAM. Move what we have to a urlpath-backed array and keep
        going there, which bounds memory at the cost of speed.

        Deliberately NOT memory mapped. Memory-mapped writing measured ~2x faster than this on an
        NFS-mounted cluster filesystem, but a shared writable mapping over NFS has no reliable
        write-back ordering guarantees, and blosc2 needs an initial_mapping_size >= the final file
        size, which we do not know while still adding cases. Correctness beats 2x on a path that
        only datasets roughly five times the size of TotalSegmentator ever reach.
        """
        if self._on_disk:
            return
        print(f'INFO: the foreground sampling location store exceeded {self._max_in_memory_bytes / 1024 ** 3:.1f} '
              f'GiB compressed and is being written incrementally from here on. This is a lot slower on '
              f'network filesystems.')
        blosc2.set_nthreads(self._nthreads)
        disk = blosc2.zeros(shape=(self._written,), dtype=np.uint64, chunks=(self.chunk_size,),
                            blocks=(self.block_size,), urlpath=self._locations_path,
                            cparams=_cparams(self._clevel, self._nthreads))
        step = self.chunk_size * 32
        for start in range(0, self._written, step):
            stop = min(start + step, self._written)
            disk[start:stop] = self._array[start:stop]
        del self._array
        self._array = disk
        self._on_disk = True

    def _flush(self, aligned_only: bool):
        if self._total == self._written:
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
        if not self._on_disk and self._array.schunk.cbytes > self._max_in_memory_bytes:
            self._spill_to_disk()

    def finalize(self):
        if self._finalized:
            return
        self._flush(aligned_only=False)
        assert self._written == self._total, (self._written, self._total)

        if not self._on_disk:
            # one sequential write instead of per-chunk filesystem traffic
            with open(self._locations_path, 'wb') as f:
                f.write(self._array.to_cframe())
                f.flush()
                os.fsync(f.fileno())

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
