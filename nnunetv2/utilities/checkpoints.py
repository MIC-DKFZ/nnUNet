"""
Reading and writing nnU-Net checkpoints.

nnU-Net writes its checkpoints as safetensors files (`checkpoint_final.safetensors`, `checkpoint_best.safetensors`,
`checkpoint_latest.safetensors`). Checkpoints written by older versions are `torch.save` pickles (`.pth`), and these
are still read, exactly as before.

Why not `.pth`: loading a pickle (`torch.load(..., weights_only=False)`) runs whatever code the file asks for.
safetensors is a JSON header followed by raw tensor bytes, so reading one cannot run code.

A checkpoint is a dict (network weights, optimizer state, logging, init args, ...). In the safetensors file:
- every tensor is stored as a tensor. The network weights keep their state dict names; all other tensors are named
  after their position in the checkpoint, e.g. `optimizer_state/state/0/momentum_buffer`.
- tensors that share memory are stored once. dynamic_network_architectures' UNets list each tensor under up to four
  names in their state dict (`...conv.weight` is also `...all_modules.0.weight`, the encoder appears again under
  `decoder.encoder.`), and storing every name separately would make the file 3-4x larger.
- everything that is not a tensor is stored as JSON in the uint8 tensor `__nnunet_checkpoint_state__`, with tensors
  replaced by references to their stored names. Python types JSON has no notion of (tuples, numpy scalars, dicts
  with non-string keys, torch.device, ...) are tagged, so loading gives back the same dict, types included, and the
  attributes of OrderedDicts are kept (state_dict() stores the modules' versions in `_metadata`). The JSON
  is not put into the safetensors header because that is capped at 100 MB and the per-class Dice history of a long
  training with many classes can get close.
- the header metadata repeats a few fields for people and tools that only read the header (trainer name,
  configuration, fold, epoch, ...). Loading does not use them.

Only plain data can be stored: tensors, numpy scalars and arrays, None, bool, int, float, str, lists, tuples, dicts,
torch.device and torch.dtype. Everything nnU-Net itself puts into a checkpoint is covered. Saving anything else
(custom objects, a module's non-tensor `get_extra_state()`, sparse or quantized tensors) raises a ValueError that
names the offending entries.
"""
import importlib.metadata
import json
import os
import struct
import sys
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open

CHECKPOINT_EXTENSION = '.safetensors'
LEGACY_CHECKPOINT_EXTENSION = '.pth'

FORMAT_NAME = 'nnunetv2-checkpoint'
FORMAT_VERSION = 1

# only needed to continue a training. checkpoint_final does not keep them: training never continues from it, and
# the optimizer state is as large as the network weights
TRAINING_STATE_KEYS = ('optimizer_state', 'grad_scaler_state')
# for trainers whose load_checkpoint gets a checkpoint without optimizer state while epochs remain
NO_OPTIMIZER_STATE_WARNING = ('WARNING: this checkpoint has no optimizer state (checkpoint_final does not keep it). '
                              'If training continues from here, the optimizer starts from scratch.')

_STATE_TENSOR = '__nnunet_checkpoint_state__'
_TAG = '$type'
# tensor names that cannot be used for stored tensors: our JSON state, and the key safetensors keeps its header
# metadata under
_RESERVED_NAMES = (_STATE_TENSOR, '__metadata__')

_DTYPES = {
    torch.float64: 'F64', torch.float32: 'F32', torch.float16: 'F16', torch.bfloat16: 'BF16',
    torch.int64: 'I64', torch.int32: 'I32', torch.int16: 'I16', torch.int8: 'I8', torch.uint8: 'U8',
    torch.bool: 'BOOL', torch.complex64: 'C64',
}
for _name, _code in (('float8_e4m3fn', 'F8_E4M3'), ('float8_e5m2', 'F8_E5M2'),
                     ('uint16', 'U16'), ('uint32', 'U32'), ('uint64', 'U64')):
    if hasattr(torch, _name):  # these dtypes are not in every torch version we support
        _DTYPES[getattr(torch, _name)] = _code


def other_format_filename(filename: str) -> Optional[str]:
    """`x.safetensors` -> `x.pth` and the other way round, None for any other extension."""
    if filename.endswith(CHECKPOINT_EXTENSION):
        return filename[:-len(CHECKPOINT_EXTENSION)] + LEGACY_CHECKPOINT_EXTENSION
    if filename.endswith(LEGACY_CHECKPOINT_EXTENSION):
        return filename[:-len(LEGACY_CHECKPOINT_EXTENSION)] + CHECKPOINT_EXTENSION
    return None


def safetensors_filename(filename: str) -> str:
    """
    The name a checkpoint is written under: `x.pth` -> `x.safetensors`, `x.safetensors` stays, anything else gets
    `.safetensors` appended.
    """
    if filename.endswith(CHECKPOINT_EXTENSION):
        return filename
    if filename.endswith(LEGACY_CHECKPOINT_EXTENSION):
        return filename[:-len(LEGACY_CHECKPOINT_EXTENSION)] + CHECKPOINT_EXTENSION
    return filename + CHECKPOINT_EXTENSION


def find_checkpoint_file(filename: str) -> Optional[str]:
    """
    A checkpoint is identified by its name without the extension. Returns `filename` if it exists, else the same
    name in the other format (`.pth` <-> `.safetensors`) if that exists, else None. So `checkpoint_final.safetensors`
    finds the `checkpoint_final.pth` of a model trained with an older nnU-Net, and the other way round.
    """
    if os.path.isfile(filename):
        other = other_format_filename(filename)
        if other is not None and os.path.isfile(other):
            warnings.warn(f'Both {filename} and {other} exist. Using {filename}. To use the other one, pass '
                          f'{os.path.basename(other)} where a checkpoint name is taken (nnUNetv2_predict -chk), or '
                          f'move {os.path.basename(filename)} out of the folder. nnUNetv2_train (--c, --val, '
                          f'--val_best) always uses the .safetensors file.')
        return filename
    other = other_format_filename(filename)
    if other is not None and os.path.isfile(other):
        return other
    return None


def resolve_checkpoint_file(filename: str) -> str:
    """find_checkpoint_file, but raises FileNotFoundError naming every file it looked for."""
    found = find_checkpoint_file(filename)
    if found is None:
        other = other_format_filename(filename)
        tried = filename if other is None else f'{filename} or {other}'
        raise FileNotFoundError(f'Checkpoint not found: {tried}')
    return found


def remove_checkpoint_files(filename: str) -> None:
    """Deletes the checkpoint in both formats, if present."""
    for f in (filename, other_format_filename(filename)):
        if f is not None and os.path.isfile(f):
            os.remove(f)


def without_training_state(checkpoint: dict) -> dict:
    """The checkpoint without what is only needed to continue a training (see TRAINING_STATE_KEYS)."""
    return {k: v for k, v in checkpoint.items() if k not in TRAINING_STATE_KEYS}


class _Encoder:
    """Turns a checkpoint into JSON-able data plus the tensors it references."""

    def __init__(self):
        self.tensors: Dict[str, torch.Tensor] = {}
        self._name_of_view: Dict[tuple, str] = {}
        self.unsupported: List[str] = []

    def _tensor(self, t: torch.Tensor, path: tuple):
        if type(t) not in (torch.Tensor, torch.nn.Parameter) or t.layout != torch.strided or t.is_quantized or \
                t.device.type == 'meta' or t.dtype not in _DTYPES:
            self.unsupported.append(f'{_path_str(path)}: {type(t).__name__} {t.dtype} {t.layout} on {t.device}')
            return None
        # exact aliases (same memory, same view) are stored once. Two tensors that only partly overlap are stored
        # separately, each with its own values. Empty tensors may all have data_ptr 0 and get aliased to each other
        # if shape, stride and dtype match, which is harmless: they hold no data.
        # a conjugate or negative view has the same memory and strides as the tensor it views, but other values
        view = (t.device, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()),
                t.dtype, t.is_conj(), t.is_neg())
        if view not in self._name_of_view:
            if len(path) == 2 and path[0] == 'network_weights':
                name = str(path[1])
            else:
                name = '/'.join(str(p) for p in path)
            unique, i = name, 1
            while unique in self.tensors or unique in _RESERVED_NAMES:
                i += 1
                unique = f'{name}#{i}'
            self._name_of_view[view] = unique
            self.tensors[unique] = t
        return {_TAG: 'tensor', 'name': self._name_of_view[view]}

    def encode(self, x: Any, path: tuple = ()):
        t = type(x)
        if x is None or t in (bool, int, float, str):
            return x
        if isinstance(x, torch.Tensor):
            return self._tensor(x, path)
        if isinstance(x, np.generic):  # before anything else: np.float64 is a subclass of float
            if not _is_plain_numpy_scalar(x):
                self.unsupported.append(f'{_path_str(path)}: numpy {x.dtype}')
                return None
            return {_TAG: 'numpy', 'dtype': x.dtype.name, 'value': x.item()}
        if t is np.ndarray:
            try:
                # reshape: ascontiguousarray turns a 0-d array into shape (1,)
                as_tensor = torch.from_numpy(np.ascontiguousarray(x).reshape(x.shape))
            except (TypeError, ValueError):  # a dtype torch does not have, or non-native byte order
                self.unsupported.append(f'{_path_str(path)}: numpy array of {x.dtype}')
                return None
            return {_TAG: 'ndarray', 'tensor': self._tensor(as_tensor, path)}
        if t is list:
            if len(x) > 0 and isinstance(x[0], np.generic) and _is_plain_numpy_scalar(x[0]) and \
                    all(type(v) is type(x[0]) for v in x):
                # the logger's histories (per-class Dice of every epoch, ...): one tag for the whole list instead of
                # one per value, which would make them about three times larger
                return {_TAG: 'numpy_list', 'dtype': x[0].dtype.name, 'values': [v.item() for v in x]}
            return [self.encode(v, path + (i,)) for i, v in enumerate(x)]
        if t is tuple:
            return {_TAG: 'tuple', 'items': [self.encode(v, path + (i,)) for i, v in enumerate(x)]}
        if t in (dict, OrderedDict):
            if t is dict and _TAG not in x and all(type(k) is str for k in x.keys()):
                return {k: self.encode(v, path + (k,)) for k, v in x.items()}
            encoded = {_TAG: 'OrderedDict' if t is OrderedDict else 'dict',
                       'items': [[self.encode(k, path + ('<key>',)), self.encode(v, path + (k,))] for k, v in x.items()]}
            if t is OrderedDict and vars(x):
                # state_dict() stores the modules' versions in the attribute _metadata, and load_state_dict passes
                # them on to the modules
                encoded['attributes'] = self.encode(vars(x), path + ('<attributes>',))
            return encoded
        if t is torch.device:
            return {_TAG: 'torch.device', 'value': str(x)}
        if t is torch.dtype:
            return {_TAG: 'torch.dtype', 'value': str(x).split('.')[-1]}
        self.unsupported.append(f'{_path_str(path)}: {t.__module__}.{t.__qualname__}')
        return None


def _is_plain_numpy_scalar(x: np.generic) -> bool:
    # bool, int, uint, float of at most 64 bit: item() gives a Python value that converts back exactly
    return x.dtype.kind in 'biuf' and x.dtype.itemsize <= 8


def _path_str(path: tuple) -> str:
    return '/'.join(str(p) for p in path) if path else '<root>'


def _decode(obj: Any, get_tensor: Callable[[str], torch.Tensor]):
    if isinstance(obj, list):
        return [_decode(v, get_tensor) for v in obj]
    if not isinstance(obj, dict):
        return obj
    tag = obj.get(_TAG)
    if tag is None:
        return {k: _decode(v, get_tensor) for k, v in obj.items()}
    if tag == 'tensor':
        return get_tensor(obj['name'])
    if tag == 'tuple':
        return tuple(_decode(v, get_tensor) for v in obj['items'])
    if tag in ('dict', 'OrderedDict'):
        items = [(_decode(k, get_tensor), _decode(v, get_tensor)) for k, v in obj['items']]
        if tag == 'dict':
            return dict(items)
        result = OrderedDict(items)
        for k, v in _decode(obj.get('attributes', {}), get_tensor).items():
            setattr(result, k, v)
        return result
    if tag in ('numpy', 'numpy_list'):
        dtype = np.dtype(obj['dtype'])
        if dtype.kind not in 'biuf':
            raise ValueError(f'Unsupported numpy dtype in checkpoint: {dtype}')
        if tag == 'numpy':
            return dtype.type(obj['value'])
        return [dtype.type(v) for v in obj['values']]
    if tag == 'ndarray':
        return _decode(obj['tensor'], get_tensor).cpu().numpy()
    if tag == 'torch.device':
        return torch.device(obj['value'])
    if tag == 'torch.dtype':
        dtype = getattr(torch, obj['value'], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f'Unknown torch dtype in checkpoint: {obj["value"]}')
        return dtype
    raise ValueError(f'Unknown type tag in checkpoint: {tag}')


def _write_safetensors(tensors: Dict[str, torch.Tensor], metadata: Dict[str, str], filename: str) -> None:
    """
    Writes the safetensors format (https://github.com/huggingface/safetensors#format): 8 bytes header length, the
    JSON header, then the raw bytes of every tensor, back to back. We write it ourselves instead of calling
    safetensors.torch.save_file because that needs every tensor on the CPU at the same time, which for a large
    model's training state on GPU means several GB of extra RAM per save. Here one tensor is copied at a time, like
    torch.save does.
    """
    if sys.byteorder != 'little':
        raise RuntimeError('safetensors files are little endian; writing them on a big endian machine is not '
                           'supported')
    # largest element size first (then by name), so that every tensor starts at an offset aligned to its element
    # size. This is the order the safetensors library writes too.
    order = sorted(tensors.keys(), key=lambda k: (-tensors[k].element_size(), k))
    header = {}
    offset = 0
    for k in order:
        t = tensors[k]
        nbytes = t.numel() * t.element_size()
        header[k] = {'dtype': _DTYPES[t.dtype], 'shape': list(t.shape), 'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    header['__metadata__'] = metadata
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_bytes += b' ' * (-len(header_bytes) % 8)

    with open(filename, 'wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        for k in order:
            if tensors[k].numel() == 0:
                continue
            f.write(_as_bytes(tensors[k]).numpy().data)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """The tensor's values as a flat uint8 tensor on the CPU, in row-major order."""
    # resolve: a conjugate or negative view holds other values than its memory, and cannot be viewed as bytes
    return t.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)


def _encode_checkpoint(checkpoint: dict) -> Tuple[_Encoder, Any]:
    if not isinstance(checkpoint, dict) or not all(type(k) is str for k in checkpoint.keys()):
        raise ValueError('A checkpoint must be a dict with string keys')
    encoder = _Encoder()
    state = encoder.encode(checkpoint)
    if encoder.unsupported:
        raise ValueError('This checkpoint cannot be saved as safetensors because these entries are neither tensors '
                         'nor plain data (None, bool, int, float, str, list, tuple, dict, numpy scalars and arrays, '
                         'torch.device, torch.dtype):\n' + '\n'.join(encoder.unsupported) + '\n'
                         'Entries ending in _extra_state come from modules that implement get_extra_state().')
    return encoder, state


def check_checkpoint_storable(checkpoint: dict) -> None:
    """
    Raises the ValueError that save_checkpoint_file would raise for `checkpoint`, without writing anything. Cheap:
    tensors are only looked at, not copied.
    """
    _encode_checkpoint(checkpoint)


def save_checkpoint_file(checkpoint: dict, filename: str) -> None:
    """
    Writes `checkpoint` as a safetensors file. `filename` must end with `.safetensors`.

    The file is written to `filename + '.tmp'` and then renamed, so a process killed while writing never leaves a
    truncated checkpoint behind.

    Raises ValueError, before anything is written, if the checkpoint holds something that cannot be stored (see the
    module docstring); the message names every such entry.
    """
    if not filename.endswith(CHECKPOINT_EXTENSION):
        raise ValueError(f'Checkpoints are written as {CHECKPOINT_EXTENSION} files, got {filename}')
    encoder, state = _encode_checkpoint(checkpoint)
    tensors = dict(encoder.tensors)
    tensors[_STATE_TENSOR] = torch.frombuffer(bytearray(json.dumps(state).encode('utf-8')), dtype=torch.uint8)

    init_args = checkpoint.get('init_args')
    init_args = init_args if isinstance(init_args, dict) else {}
    metadata = {
        'format': FORMAT_NAME,
        'format_version': FORMAT_VERSION,
        'nnunetv2_version': _nnunet_version(),
        'trainer_name': checkpoint.get('trainer_name'),
        'configuration': init_args.get('configuration'),
        'fold': init_args.get('fold'),
        'current_epoch': checkpoint.get('current_epoch'),
        'inference_allowed_mirroring_axes': checkpoint.get('inference_allowed_mirroring_axes'),
        'contains_training_state': all(k in checkpoint for k in TRAINING_STATE_KEYS),
    }
    # safetensors only allows strings in the header metadata
    metadata = {k: json.dumps(_plain(v)) for k, v in metadata.items()}

    tmp = filename + '.tmp'
    try:
        _write_safetensors(tensors, metadata, tmp)
        os.replace(tmp, filename)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)


def _nnunet_version() -> Optional[str]:
    try:
        return importlib.metadata.version('nnunetv2')
    except importlib.metadata.PackageNotFoundError:
        return None


def _plain(x):
    # for the informational header fields only
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [_plain(i) for i in x]
    if x is None or type(x) in (bool, int, float, str):
        return x
    return str(x)


def _read_header_metadata(f) -> Dict[str, Any]:
    # our own entries are JSON. Other writers put plain strings there (transformers and timm write
    # {"format": "pt"}); those are returned as they are
    metadata = {}
    for k, v in (f.metadata() or {}).items():
        try:
            metadata[k] = json.loads(v)
        except json.JSONDecodeError:
            metadata[k] = v
    return metadata


def _open_safetensors(filename: str):
    try:
        return safe_open(filename, framework='pt', device='cpu')
    except Exception:
        with open(filename, 'rb') as fh:
            magic = fh.read(4)
        if magic.startswith(b'PK\x03\x04') or magic.startswith(b'\x80'):
            raise ValueError(f'{filename} is not a safetensors file but a torch.save pickle with a .safetensors name, '
                             f'probably written by a custom trainer that overrides save_checkpoint and calls '
                             f'torch.save. It is not loaded, because loading a pickle runs code. If you trust the '
                             f'file, rename it to {LEGACY_CHECKPOINT_EXTENSION} and it is loaded the way older '
                             f'checkpoints are.') from None
        raise


def _device_str(map_location) -> str:
    if map_location is None:
        return 'cpu'
    if isinstance(map_location, (str, torch.device)):
        return str(torch.device(map_location))
    raise ValueError(f'map_location must be None, a str or a torch.device for safetensors checkpoints, got '
                     f'{map_location!r}')


def _owned(t: torch.Tensor, device: str) -> torch.Tensor:
    """
    safetensors maps the file into memory, and a CPU tensor it returns reads from that mapping for as long as it
    lives: if the file is overwritten in place the tensor's values change, and if it is truncated or sits on a network
    mount that drops, the process dies (SIGBUS). Windows cannot delete or replace a mapped file at all. So copy, like
    torch.load gives tensors that own their memory. Copying to a GPU does that anyway.
    """
    return t.clone() if torch.device(device).type == 'cpu' else t.to(device)


def _read_checkpoint(f, filename: str, device: str, keys: Optional[Iterable[str]]) -> dict:
    metadata = _read_header_metadata(f)
    if metadata.get('format') != FORMAT_NAME:
        raise ValueError(f'{filename} is a safetensors file but not an nnU-Net checkpoint (it has no '
                         f'"format": "{FORMAT_NAME}" header entry). Plain network weights can be used with '
                         f'-pretrained_weights.')
    if metadata.get('format_version', 0) > FORMAT_VERSION:
        raise ValueError(f'{filename} was written by a newer nnU-Net (checkpoint format version '
                         f'{metadata["format_version"]}, this version reads up to {FORMAT_VERSION}). Please '
                         f'update nnU-Net.')
    state = json.loads(bytes(f.get_tensor(_STATE_TENSOR).numpy()).decode('utf-8'))

    cache: Dict[str, torch.Tensor] = {}

    def get_tensor(name: str) -> torch.Tensor:
        # aliases decode to the same tensor object, like torch.load gives them back as views of one storage
        if name not in cache:
            cache[name] = _owned(f.get_tensor(name), device)
        return cache[name]

    if keys is None:
        return _decode(state, get_tensor)
    if _TAG in state:
        raise ValueError(f'{filename}: unexpected checkpoint layout')
    return {k: _decode(state[k], get_tensor) for k in keys if k in state}


def load_checkpoint_file(filename: str, map_location: Union[str, torch.device, None] = 'cpu',
                         keys: Optional[Iterable[str]] = None) -> dict:
    """
    Loads a checkpoint. `filename` may name either format: if it does not exist, the same name with the other
    extension is loaded (see find_checkpoint_file). `.safetensors` files are read as described in the module
    docstring. `.pth` files are loaded the way nnU-Net always has: torch.load(filename, map_location=map_location,
    weights_only=False). That unpickles the file and therefore runs any code in it, so only load .pth files you trust.

    keys: for .safetensors files, only these top level entries are loaded (missing ones are skipped). Inference uses
    this to skip the optimizer state of checkpoint_best/latest. Ignored for .pth files, which are always loaded in
    full.
    """
    filename = resolve_checkpoint_file(filename)
    if filename.endswith(CHECKPOINT_EXTENSION):
        with _open_safetensors(filename) as f:
            return _read_checkpoint(f, filename, _device_str(map_location), keys)
    return torch.load(filename, map_location=map_location, weights_only=False)


def load_network_weights(filename: str, map_location: Union[str, torch.device, None] = 'cpu') -> dict:
    """
    The network weights (a state dict) from an nnU-Net checkpoint in either format, or from a safetensors file that
    holds nothing but a state dict (for example weights trained outside nnU-Net). Names resolve like in
    load_checkpoint_file.
    """
    filename = resolve_checkpoint_file(filename)
    if not filename.endswith(CHECKPOINT_EXTENSION):
        return load_checkpoint_file(filename, map_location)['network_weights']
    device = _device_str(map_location)
    with _open_safetensors(filename) as f:
        if _read_header_metadata(f).get('format') != FORMAT_NAME:
            return OrderedDict((k, _owned(f.get_tensor(k), device)) for k in f.keys())
        return _read_checkpoint(f, filename, device, ('network_weights',))['network_weights']


def read_checkpoint_metadata(filename: str) -> Dict[str, Any]:
    """The informational header fields of a .safetensors checkpoint (trainer_name, configuration, fold, ...)."""
    with _open_safetensors(filename) as f:
        return _read_header_metadata(f)


def assert_checkpoints_equal(a: Any, b: Any, path: tuple = ()) -> None:
    """
    Raises AssertionError naming the first difference. Tensors must match bit for bit (dtype, shape and bytes;
    devices may differ), NaN equals NaN, and types must match: a numpy float32 is not a float, a tuple not a list.
    torch.nn.Parameter and torch.Tensor count as the same type. The attributes of OrderedDicts are compared too.

    No assert statements: python -O removes them, and the converter deletes the .pth once this passes.
    """
    where = _path_str(path)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            raise AssertionError(f'{where}: {type(a)} vs {type(b)}')
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f'{where}: {a.dtype} {a.shape} vs {b.dtype} {b.shape}')
        if not torch.equal(_as_bytes(a), _as_bytes(b)):
            raise AssertionError(f'{where}: tensor values differ')
        return
    if type(a) is not type(b):
        raise AssertionError(f'{where}: {type(a)} vs {type(b)}')
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f'{where}: {a.dtype} {a.shape} vs {b.dtype} {b.shape}')
        if a.tobytes() != b.tobytes():
            raise AssertionError(f'{where}: array values differ')
    elif isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            raise AssertionError(f'{where}: keys differ')
        for k in a.keys():
            assert_checkpoints_equal(a[k], b[k], path + (k,))
        if isinstance(a, OrderedDict):
            assert_checkpoints_equal(vars(a), vars(b), path + ('<attributes>',))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f'{where}: length {len(a)} vs {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            assert_checkpoints_equal(x, y, path + (i,))
    elif isinstance(a, (float, np.floating)) and np.isnan(a):
        if not np.isnan(b):
            raise AssertionError(f'{where}: {a} vs {b}')
    elif a != b:
        raise AssertionError(f'{where}: {a!r} vs {b!r}')
