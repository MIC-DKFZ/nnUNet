"""
Shared plumbing for nnU-Net's multiprocessing passes: hand one task per item to a spawn pool and
consume the results as they land, while watching for workers that quietly died (usually OOM).
"""
import multiprocessing
from time import sleep
from typing import Any, Callable, Iterator, Sequence, Tuple

from tqdm import tqdm

WORKER_DIED_MESSAGE = ('Some background worker is 6 feet under. Yuck. \n'
                       'OK jokes aside.\n'
                       'One of your background processes is missing. This could be because of '
                       'an error (look for an error message) or because it was killed '
                       'by your OS due to running out of RAM. If you don\'t see '
                       'an error message, out of RAM is likely the problem. In that case '
                       'reducing the number of workers might help')


def imap_unordered_with_progress(fn: Callable, arg_tuples: Sequence[Sequence], num_processes: int,
                                 desc: str = None, show_progress_bar: bool = True) -> Iterator[Tuple[int, Any]]:
    """
    Runs ``fn(*args)`` for every entry of `arg_tuples` in a spawn pool and yields ``(index, result)``
    in completion order, where `index` is the position in `arg_tuples`. Each result is dropped as
    soon as it has been yielded, so a consumer that streams results to disk never accumulates them.

    Exceptions raised in a worker surface here (via ``.get()``); a worker that vanished without
    raising - which is what an OOM kill looks like - raises RuntimeError.

    This is a generator: it must be consumed to completion for the pool to be torn down cleanly.
    """
    with multiprocessing.get_context('spawn').Pool(num_processes) as p:
        # p is pretty nifti. If we kill workers they just respawn but don't do any work.
        # So we need to store the original pool of workers.
        workers = list(p._pool)
        results = [p.apply_async(fn, args) for args in arg_tuples]
        remaining = len(results)
        with tqdm(desc=desc, total=len(results), disable=not show_progress_bar) as pbar:
            while remaining > 0:
                if not all(j.is_alive() for j in workers):
                    raise RuntimeError(WORKER_DIED_MESSAGE)
                progressed = False
                for i, r in enumerate(results):
                    if r is not None and r.ready():
                        yield i, r.get()
                        results[i] = None
                        remaining -= 1
                        pbar.update()
                        progressed = True
                if not progressed:
                    sleep(0.1)
