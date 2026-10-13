import os
from typing import Iterable, List, Optional

import torch

from nnunetv2.utilities.checkpoints import CHECKPOINT_EXTENSION, LEGACY_CHECKPOINT_EXTENSION, \
    assert_checkpoints_equal, load_checkpoint_file, safetensors_filename, save_checkpoint_file, without_training_state


def convert_checkpoint(pth_file: str, keep_pth: bool = False, overwrite: bool = False) -> Optional[str]:
    """
    Converts one .pth checkpoint to .safetensors next to it. Returns the new file, or None if it already existed and
    overwrite is False.

    checkpoint_final.pth loses its optimizer and grad scaler state on the way, like the checkpoint_final nnU-Net
    writes now. Everything else is kept. The new file is written under a temporary name, read back and compared with
    the .pth (every tensor bit for bit, everything else including its type), and only then renamed to its final
    name. After that the .pth is deleted, unless keep_pth is True. If anything goes wrong, the .pth stays, and so
    does a .safetensors that existed before (overwrite=True); the new file is not left behind.

    Loading a .pth runs any code in it. Only convert checkpoints you trust.
    """
    if not pth_file.endswith(LEGACY_CHECKPOINT_EXTENSION):
        raise ValueError(f'Expected a {LEGACY_CHECKPOINT_EXTENSION} file, got {pth_file}')
    target = safetensors_filename(pth_file)
    if os.path.isfile(target) and not overwrite:
        return None

    try:
        # mmap: a checkpoint can be several GB and we only need to stream it once
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        # mmap needs the zip format torch.save writes since torch 1.6
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
    if os.path.basename(pth_file) == 'checkpoint_final.pth':
        checkpoint = without_training_state(checkpoint)

    # Not written as `target` directly, for two reasons. Nothing that loads checkpoint_final.safetensors must find an
    # unverified file and take it over the .pth. And load_checkpoint_file falls back to the other format when the
    # named file is missing: had `target` vanished before the read-back (another process), the .pth would have been
    # compared with itself and then deleted. This name has no counterpart in the other format.
    unverified = target[:-len(CHECKPOINT_EXTENSION)] + f'.unverified{os.getpid()}' + CHECKPOINT_EXTENSION
    save_checkpoint_file(checkpoint, unverified)
    try:
        reloaded = load_checkpoint_file(unverified)
        try:
            assert_checkpoints_equal(checkpoint, reloaded)
        except AssertionError as e:
            raise RuntimeError(f'Converting {pth_file} does not reproduce it ({e}). The .pth is unchanged.') from e
        os.replace(unverified, target)
    finally:
        if os.path.isfile(unverified):
            os.remove(unverified)

    # with mmap, the loaded tensors keep the .pth open, and Windows cannot delete an open file
    del checkpoint, reloaded
    if not keep_pth:
        os.remove(pth_file)
    return target


def _exists_in_either_format(pth_file: str) -> bool:
    # not find_checkpoint_file: that warns when both formats exist, which --keep_pth leaves behind on purpose
    return os.path.isfile(pth_file) or os.path.isfile(safetensors_filename(pth_file))


def find_checkpoints_to_convert(folders: Iterable[str], checkpoint_names: Iterable[str]) -> List[str]:
    """
    The .pth files with the given names in finished trainings below `folders`. A training is finished if it has a
    checkpoint_final and no checkpoint_latest (in either format).
    """
    found = []
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            candidates = [os.path.join(dirpath, n) for n in checkpoint_names if n in filenames]
            if not candidates:
                continue
            finished = _exists_in_either_format(os.path.join(dirpath, 'checkpoint_final.pth')) and \
                not _exists_in_either_format(os.path.join(dirpath, 'checkpoint_latest.pth'))
            if not finished:
                print(f'Skipping {dirpath}: the training is not finished (no checkpoint_final, or a checkpoint_latest '
                      f'exists). Continue it with nnUNetv2_train ... --c instead; from then on it writes '
                      f'.safetensors checkpoints and removes the .pth ones it replaces.')
                continue
            found.extend(candidates)
    return found


def convert_checkpoints_entry():
    import argparse
    parser = argparse.ArgumentParser(
        description='Converts the .pth checkpoints of models trained with older nnU-Net versions to .safetensors, '
                    'the format nnU-Net writes now. Searches the given folders recursively and converts every '
                    'finished training it finds (one with a checkpoint_final and no checkpoint_latest). Each new file '
                    'is read back and compared with the .pth before the .pth is deleted. checkpoint_final loses its '
                    'optimizer state, like the checkpoint_final nnU-Net writes now; all other checkpoints are '
                    'converted in full. WARNING: loading a .pth file runs any code it contains. Only convert '
                    'checkpoints you trust. Once the .pth files are deleted, nnU-Net versions that cannot read '
                    '.safetensors cannot use the model anymore.')
    parser.add_argument('-i', type=str, nargs='+', required=True,
                        help='Folders to search, for example a trained model folder (Trainer__Plans__config), a single '
                             'fold_X, a dataset folder or all of nnUNet_results.')
    parser.add_argument('-chk', type=str, nargs='+', required=False,
                        default=('checkpoint_final.pth', 'checkpoint_best.pth'),
                        help='Names of the checkpoints to convert. Default: checkpoint_final.pth checkpoint_best.pth')
    parser.add_argument('--keep_pth', action='store_true', required=False,
                        help='Keep the .pth files. By default they are deleted once the conversion is verified.')
    parser.add_argument('--overwrite', action='store_true', required=False,
                        help='Convert again if the .safetensors file already exists. By default such checkpoints '
                             'are skipped and their .pth is left alone.')
    args = parser.parse_args()

    for n in args.chk:
        if not n.endswith(LEGACY_CHECKPOINT_EXTENSION):
            parser.error(f'-chk takes {LEGACY_CHECKPOINT_EXTENSION} names, got {n}')
    for folder in args.i:
        if not os.path.isdir(folder):
            parser.error(f'-i: {folder} is not a folder')
    pth_files = find_checkpoints_to_convert(args.i, args.chk)
    converted, skipped, failed = 0, 0, []
    for f in pth_files:
        size_before = os.path.getsize(f)
        try:
            target = convert_checkpoint(f, keep_pth=args.keep_pth, overwrite=args.overwrite)
        except Exception as e:
            print(f'FAILED {f}: {e}')
            failed.append(f)
            continue
        if target is None:
            skipped += 1
            print(f'Skipping {f}: {safetensors_filename(f)} exists (use --overwrite to convert again)')
        else:
            converted += 1
            print(f'{f} ({size_before / 1e6:.1f} MB) -> {os.path.basename(target)} '
                  f'({os.path.getsize(target) / 1e6:.1f} MB), verified'
                  f'{"" if args.keep_pth else ", .pth deleted"}')
    print(f'{len(pth_files)} checkpoints found: {converted} converted, {skipped} skipped, {len(failed)} failed.')
    if failed:
        raise SystemExit(f'{len(failed)} checkpoints could not be converted; their .pth files are unchanged:\n' +
                         '\n'.join(failed))
