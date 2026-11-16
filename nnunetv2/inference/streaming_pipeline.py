"""
Batch machinery for nnUNetPredictorStreaming: persistent worker pools, the work they run, and the preprocessing
iterator that feeds the GPU.

A batch prediction overlaps three stages, as nnUNetPredictor.predict_from_files does: background workers load and
preprocess the next cases, the main process runs the fused sliding-window + resample + argmax pass on the GPU, and a
second set of background workers exports the finished segmentations. Unlike nnUNetPredictor, the worker pools belong to
the predictor and are reused across calls: a persistent predictor (one that a toolkit keeps for many prediction calls)
pays the spawn start-up of its workers (Python, torch and nnU-Net imports each) once, not per call. The pools are
created on first use and closed by nnUNetPredictorStreaming.close() (also via a context manager or when the predictor
is garbage collected).

What runs in the workers is pluggable, so tools built on nnU-Net can bring their own input and output handling without
owning any of the pipeline:

  load_fn(case) -> (image, properties)
      Runs in a preprocessing worker (or in-process for single cases). image is (c, x, y, z) in the axis convention of
      the model's reader (plans 'image_reader_writer'); properties needs at least 'spacing' and may carry any extra
      keys - they pass through preprocessing unchanged and reach export_fn. None: case is a list of image files, read
      with the plans' reader class (what nnUNetv2_predict does).
  export_fn(segmentation, properties, output, **export_kwargs) -> anything
      Runs in an export worker (or in-process for single cases). segmentation is the finished label map in the input
      image's geometry and axis convention, output the per-case output spec passed in by the caller (for nnU-Net's own
      writer the truncated output file name). The return values are collected and returned in input order.

Both must be module-level functions (they are pickled by reference to reach the spawned workers), and cases, outputs
and export_kwargs must be picklable.

Failures: by default any exception aborts the batch (on_error='raise'). With on_error='skip' a case whose loading,
prediction or export fails becomes a CaseFailure in the results and the batch carries on - except when a worker process
dies (usually: killed by the OS for RAM), which always raises.
"""
import inspect
import multiprocessing
import time
import traceback
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import skip_seg_kwargs
from nnunetv2.preprocessing.resampling.utils import recursive_find_resampling_fn_by_name
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

WORKER_DIED_MESSAGE = ('A background worker of the streaming predictor died. It was likely killed by your OS because '
                       'of insufficient CPU RAM. Use fewer workers or get more RAM.')

# per-process state of a preprocessing worker, set by _init_preprocessing_worker
_worker_state = {}


@dataclass
class CaseFailure:
    """Result of a case that failed under on_error='skip': its index in the call, the stage that failed ('load' =
    load_fn + preprocessing, 'predict', 'export'), the exception and its traceback (as text: picklable)."""
    index: int
    stage: str
    error: str
    traceback: str

    @staticmethod
    def from_exception(index: int, stage: str, e: BaseException) -> 'CaseFailure':
        return CaseFailure(index, stage, repr(e), ''.join(traceback.format_exception(type(e), e, e.__traceback__)))


# ------------------------------------------------------------------ load / preprocess ------------------------------
def load_npy_case(case):
    """load_fn for in-memory cases: case is (image, properties). The properties are copied, the caller's stay intact."""
    image, properties = case
    return image, deepcopy(properties)


def with_resampling_threads(configuration_manager: ConfigurationManager, num_threads: int) -> ConfigurationManager:
    """configuration_manager, or a copy whose data resampling uses num_threads CPU threads - if its resampling function
    takes num_threads (torch resampling does) and the plans do not set it themselves. More threads give the same
    result (each output voxel is computed independently)."""
    kwargs = configuration_manager.configuration['resampling_fn_data_kwargs']
    if 'num_threads' in kwargs:
        return configuration_manager
    fn = recursive_find_resampling_fn_by_name(configuration_manager.configuration['resampling_fn_data'])
    if 'num_threads' not in inspect.signature(fn).parameters:
        return configuration_manager
    return ConfigurationManager({**configuration_manager.configuration,
                                 'resampling_fn_data_kwargs': {**kwargs, 'num_threads': num_threads}})


def load_and_preprocess(load_fn: Optional[Callable], case, plans_manager: PlansManager,
                        configuration_manager: ConfigurationManager, dataset_json: dict, preprocessor,
                        resampling_threads: Optional[int] = None):
    """Load one case (load_fn, or the plans' reader for a list of files: the preprocessor's run_case, as in
    nnUNetv2_predict) and run nnU-Net's preprocessing on it, without the nonzero-mask segmentation inference does not
    use (skip_seg_kwargs). resampling_threads: CPU threads for torch resampling (with_resampling_threads; None: the
    plans' setting). Returns (data float32 (c, *internal shape), properties)."""
    if resampling_threads is not None:
        configuration_manager = with_resampling_threads(configuration_manager, resampling_threads)
    if load_fn is None:
        data, _, properties = preprocessor.run_case(case, None, plans_manager, configuration_manager, dataset_json,
                                                    **skip_seg_kwargs(preprocessor.run_case))
    else:
        image, properties = load_fn(case)
        data, _, properties = preprocessor.run_case_npy(image, None, properties, plans_manager, configuration_manager,
                                                        dataset_json, **skip_seg_kwargs(preprocessor.run_case_npy))
    return np.ascontiguousarray(data, dtype=np.float32), properties


def _init_preprocessing_worker(plans: dict, configuration: dict, dataset_json: dict, verbose: bool):
    plans_manager = PlansManager(plans)
    configuration_manager = ConfigurationManager(configuration)
    _worker_state['preprocessing'] = (plans_manager, configuration_manager, dataset_json,
                                      configuration_manager.preprocessor_class(verbose=verbose))


def _preprocess_in_worker(load_fn: Optional[Callable], case, output) -> dict:
    t = time.perf_counter()
    data, properties = load_and_preprocess(load_fn, case, *_worker_state['preprocessing'])
    return {'data': data, 'data_properties': properties, 'ofile': output,
            'timings': {'load_and_preprocess': time.perf_counter() - t}}


# ------------------------------------------------------------------ export -----------------------------------------
def write_segmentation(segmentation: np.ndarray, properties: dict, output_file_truncated: str,
                       reader_writer_class, file_ending: str) -> None:
    """Default export_fn: write the segmentation with the model's reader/writer class (what nnUNetv2_predict writes)."""
    reader_writer_class().write_seg(segmentation, output_file_truncated + file_ending, properties)


def timed_export(export_fn: Callable, segmentation, properties: dict, output, export_kwargs: dict):
    """Runs export_fn in an export worker and returns (its result, seconds)."""
    t = time.perf_counter()
    result = export_fn(segmentation, properties, output, **export_kwargs)
    return result, time.perf_counter() - t


# ------------------------------------------------------------------ pools -------------------------------------------
def new_pool(num_processes: int, initializer: Optional[Callable] = None, initargs: tuple = ()) -> ProcessPoolExecutor:
    """A spawn-context process pool, as everywhere in nnU-Net inference (fork is unsafe once CUDA is initialized)."""
    return ProcessPoolExecutor(max_workers=num_processes, mp_context=multiprocessing.get_context('spawn'),
                               initializer=initializer, initargs=initargs)


def shutdown_pools(pools: dict) -> None:
    """Shut down every pool in pools ({name: (key, ProcessPoolExecutor)}) and empty it. Used by
    nnUNetPredictorStreaming.close() and by its finalizer, which must not reference the predictor itself."""
    for name in list(pools):
        _, pool = pools.pop(name)
        pool.shutdown(wait=True, cancel_futures=True)


class PreprocessingIterator:
    """
    Iterates over cases in input order, yielding {'data', 'data_properties', 'ofile'} items preprocessed by a persistent
    worker pool. At most max_in_flight cases are submitted ahead of the consumer, so preprocessed volumes cannot pile up
    in RAM when the GPU is slower than the workers (the same bound as nnU-Net's own iterators: per worker one finished
    case waiting and one in progress). The pool is requested only when iteration starts or start() is called, so a
    caller can do other work (e.g. load model weights) while its workers spawn and preprocess. Has a length, so the
    predictor can apply its compile policy.
    """
    def __init__(self, get_pool: Callable[[], ProcessPoolExecutor], cases: Sequence[Any], load_fn: Optional[Callable],
                 outputs: Optional[Sequence[Any]], max_in_flight: int, on_broken: Callable[[], None],
                 on_error: str = 'raise'):
        assert outputs is None or len(outputs) == len(cases), 'one output per case'
        assert on_error in ('raise', 'skip'), f"on_error must be 'raise' or 'skip', got {on_error!r}"
        self.get_pool, self.cases, self.load_fn = get_pool, list(cases), load_fn
        self.outputs = list(outputs) if outputs is not None else [None] * len(self.cases)
        self.max_in_flight = max(1, max_in_flight)
        self.on_broken = on_broken
        self.on_error = on_error
        self._pool, self._pending, self._next = None, deque(), 0

    def __len__(self):
        return len(self.cases)

    def _submit(self):
        self._pending.append(self._pool.submit(_preprocess_in_worker, self.load_fn, self.cases[self._next],
                                               self.outputs[self._next]))
        self._next += 1

    def start(self):
        """Request the pool and submit the first max_in_flight cases without waiting for any of them, so the caller can
        do other work while the workers start and preprocess. Iterating calls it if the caller has not."""
        if self._pool is not None or len(self.cases) == 0:
            return
        self._pool = self.get_pool()
        while self._next < len(self.cases) and len(self._pending) < self.max_in_flight:
            self._submit()

    def cancel(self):
        """Cancel the cases submitted but not yet started (for a caller that gives up before iterating to the end)."""
        for future in self._pending:
            future.cancel()

    def __iter__(self):
        if len(self.cases) == 0:
            return
        self.start()
        pending = self._pending
        try:
            index = 0
            while pending:
                future = pending.popleft()
                try:
                    item = future.result()
                except BrokenProcessPool as e:
                    self.on_broken()
                    raise RuntimeError(WORKER_DIED_MESSAGE) from e
                except Exception as e:
                    if self.on_error == 'raise':
                        raise
                    item = {'error': CaseFailure.from_exception(index, 'load', e), 'ofile': self.outputs[index]}
                item['index'] = index
                index += 1
                if self._next < len(self.cases):
                    self._submit()
                yield item
        finally:
            self.cancel()


def collect_results(results: List[Any], on_broken: Callable[[], None], timings: Optional[List[dict]] = None,
                    on_error: str = 'raise', indices: Optional[List[int]] = None) -> List[Any]:
    """
    Results in input order. Futures (of timed_export) are waited for and unpacked - their export time goes into
    timings[i]['export'] - everything else is taken as is. With on_error='skip' a failed export becomes a CaseFailure.
    """
    out = []
    for i, r in enumerate(results):
        if hasattr(r, 'result'):
            try:
                r, seconds = r.result()
                if timings is not None:
                    timings[i]['export'] = seconds
            except BrokenProcessPool as e:
                on_broken()
                raise RuntimeError(WORKER_DIED_MESSAGE) from e
            except Exception as e:
                if on_error == 'raise':
                    raise
                r = CaseFailure.from_exception(indices[i] if indices is not None else i, 'export', e)
        out.append(r)
    return out
