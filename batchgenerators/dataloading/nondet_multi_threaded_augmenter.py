# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import traceback
from copy import deepcopy
from typing import List, Union
import threading
from builtins import range
from multiprocessing import Process
from multiprocessing import Queue
from queue import Queue as thrQueue
import numpy as np
import logging
from multiprocessing import Event
from time import sleep, time

from batchgenerators.dataloading import DataLoader
from threadpoolctl import threadpool_limits

try:
    import torch
except ImportError:
    torch = None


def producer(queue: Queue, data_loader, transform, thread_id: int, seed,
             abort_event: Event, wait_time: float = 0.02):
    # the producer will set the abort event if something happens
    np.random.seed(seed)
    data_loader.set_thread_id(thread_id)
    item = None

    try:
        while True:

            if abort_event.is_set():
                return
            else:
                if item is None:
                    item = next(data_loader)
                    if transform is not None:
                        item = transform(**item)

                if abort_event.is_set():
                    return

                if not queue.full():
                    queue.put(item)
                    item = None
                else:
                    sleep(wait_time)

    except Exception as e:
        print("Exception in background worker %d:\n" % thread_id, e)
        traceback.print_exc()
        abort_event.set()
        return


def pin_memory_of_all_eligible_items_in_dict(result_dict):
    for k in result_dict.keys():
        if isinstance(result_dict[k], torch.Tensor):
            result_dict[k] = result_dict[k].pin_memory()
    return result_dict


def results_loop(in_queue: Queue, out_queue: thrQueue, abort_event: Event,
                 pin_memory: bool, worker_list: List[Process],
                 gpu: Union[int, None] = None, wait_time: float = 0.02):
    do_pin_memory = torch is not None and pin_memory and gpu is not None and torch.cuda.is_available()

    if do_pin_memory:
        print('using pin_memory on device', gpu)
        torch.cuda.set_device(gpu)

    item = None

    while True:
        try:
            if abort_event.is_set():
                return

            # check if all workers are still alive
            if not all([i.is_alive() for i in worker_list]):
                abort_event.set()
                raise RuntimeError("Someone died. Better end this madness. This is not the actual error message! Look "
                                   "further up your "
                                   "stdout to see what caused the error. Please also check whether your RAM was full")

            if item is None:
                if not in_queue.empty():
                    item = in_queue.get()
                    if do_pin_memory:
                        item = pin_memory_of_all_eligible_items_in_dict(item)
                else:
                    sleep(wait_time)
                    continue

            # we only arrive here if item is not None. Now put item in to the out_queue
            if not out_queue.full():
                out_queue.put(item)
                item = None
            else:
                sleep(wait_time)
                continue

        except Exception as e:
            abort_event.set()
            raise e


class NonDetMultiThreadedAugmenter(object):
    """
    Non-deterministic but potentially faster than MultiThreadedAugmenter and uses less RAM. Also less complicated.
    This one only has one queue through which the communication with background workers happens, meaning that there
    can be a race condition to it (and thus a nondeterministic ordering of batches). The advantage of this approach is
    that we will never run into the issue where everything needs to wait for worker X to finish its work.
    Also this approach requires less RAM because we do not need to have some number of cached batches per worker and
    now use a global pool of caches batches that is shared among all workers.
    THIS MTA ONLY WORKS WITH DATALOADER THAT RETURN INFINITE RANDOM SAMPLES! So if you are using DataLoader, make sure
    to set infinite=True.
    Seeding this is not recommended :-)
    """

    def __init__(self, data_loader, transform, num_processes, num_cached=2, seeds=None, pin_memory=False,
                 wait_time=0.02):
        self.pin_memory = pin_memory
        self.transform = transform
        self.num_cached = num_cached

        if isinstance(data_loader, DataLoader): assert data_loader.infinite, "Only use DataLoader instances that" \
                                                                             " have infinite=True"
        self.generator = data_loader
        self.num_processes = num_processes

        self._queue = None
        self._processes = []
        self.results_loop_thread = None
        self.results_loop_queue = None
        self.abort_event = None
        self.initialized = False

        self.wait_time = wait_time

        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __get_next_item(self):
        item = None

        while item is None:
            #
            if self.abort_event.is_set():
                # self.communication_thread handles checking for dead workers and will set the abort event if necessary
                self._finish()
                raise RuntimeError("MultiThreadedAugmenter.abort_event was set, something went wrong. Maybe one of "
                                   "your workers crashed. This is not the actual error message! Look further up your "
                                   "stdout to see what caused the error. Please also check whether your RAM was full")

            if not self.results_loop_queue.empty():
                item = self.results_loop_queue.get()
                self.results_loop_queue.task_done()
            else:
                sleep(self.wait_time)

        return item

    def __next__(self):
        if not self.initialized:
            self._start()

        item = self.__get_next_item()
        return item

    def _start(self):
        if not self.initialized:
            self._finish()

            self._queue = Queue(self.num_cached)
            self.results_loop_queue = thrQueue(self.num_cached)
            self.abort_event = Event()

            logging.debug("starting workers")
            if isinstance(self.generator, DataLoader):
                self.generator.was_initialized = False

            with threadpool_limits(limits=1, user_api=None):
                for i in range(self.num_processes):
                    self._processes.append(Process(target=producer, args=(
                        self._queue, self.generator, self.transform, i, self.seeds[i], self.abort_event, self.wait_time
                    )))
                    self._processes[-1].daemon = True
            _ = [i.start() for i in self._processes]

            if torch is not None and torch.cuda.is_available():
                gpu = torch.cuda.current_device()
            else:
                gpu = None

            # in_queue: Queue, out_queue: thrQueue, abort_event: Event, pin_memory: bool, worker_list: List[Process],
            # gpu: Union[int, None] = None, wait_time: float = 0.02
            self.results_loop_thread = threading.Thread(target=results_loop, args=(
                self._queue, self.results_loop_queue, self.abort_event, self.pin_memory, self._processes, gpu,
                self.wait_time)
                                                        )
            self.results_loop_thread.daemon = True
            self.results_loop_thread.start()

            self.initialized = True
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")

    def _finish(self):
        if self.initialized:
            self.abort_event.set()
            sleep(self.wait_time)
            [i.terminate() for i in self._processes if i.is_alive()]

        del self._queue, self.results_loop_queue, self.results_loop_thread, self.abort_event, self._processes
        self._queue, self.results_loop_queue, self.results_loop_thread, self.abort_event = None, None, None, None
        self._processes = []
        self.initialized = False

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()


if __name__ == '__main__':
    from tests.test_DataLoader import DummyDataLoader
    dl = DummyDataLoader(deepcopy(list(range(1234))), 2, 3, None,
                         return_incomplete=False, shuffle=True,
                         infinite=True)

    mt = NonDetMultiThreadedAugmenter(dl, None, 3, 2, None, False, 0.02)
    mt._start()

    st = time()
    for i in range(1000):
        print(i)
        b = next(mt)
    end = time()
    print(end - st)

    mt._finish()