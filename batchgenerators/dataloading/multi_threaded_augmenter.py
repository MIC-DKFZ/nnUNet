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


from __future__ import print_function

from typing import List, Union

from future import standard_library
import threading

standard_library.install_aliases()
from builtins import range
from multiprocessing import Process
from multiprocessing import Queue
from queue import Queue as thrQueue
import numpy as np
import sys
import logging
from multiprocessing import Event
from time import sleep, time
from threadpoolctl import threadpool_limits

try:
    import torch
except ImportError:
    torch = None


def producer(queue, data_loader, transform, thread_id, seed, abort_event, wait_time: float = 0.02):
    np.random.seed(seed)
    data_loader.set_thread_id(thread_id)
    item = None

    try:
        while True:
            # check if abort event was set
            if not abort_event.is_set():
                # print("worker %d event not set" % thread_id)
                if item is None:
                    try:
                        item = next(data_loader)
                        if transform is not None:
                            item = transform(**item)
                    except StopIteration:
                        item = "end"

                if not queue.full():
                    queue.put(item)
                    item = None
                else:
                    sleep(wait_time)
            else:
                # print("worder %d event is now set, exiting" % thread_id)
                return
    except KeyboardInterrupt:
        abort_event.set()
        return
    except Exception as e:
        print(sys.last_traceback())
        print("Exception in background worker %d:\n" % thread_id, e)
        abort_event.set()
        return


def results_loop(in_queues: List[Queue], out_queue: thrQueue, abort_event: Event, pin_memory: bool,
                 gpu: Union[int, None], wait_time: float, worker_list: list):
    do_pin_memory = torch is not None and pin_memory and gpu is not None and torch.cuda.is_available()

    if do_pin_memory:
        print('using pin_memory on device', gpu)
        torch.cuda.set_device(gpu)

    item = None
    queue_ctr = 0
    end_ctr = 0

    while True:
        # if abort_event is set we need to clean up. This is where it hangs sometimes so it makes sense to drain all
        # the incoming queues and ignore all the errors occuring during this process.
        try:
            if abort_event.is_set():
                return

            # check if all workers are still alive
            if not all([i.is_alive() for i in worker_list]):
                abort_event.set()
                raise RuntimeError("Someone died. Better end this madness. This is not the actual error message! Look "
                                   "further up your "
                                   "stdout to see what caused the error. Please also check whether your RAM was full")


            # if we don't have an item we need to fetch it first. If the queue we want to get it from it empty, try
            # again later
            if item is None:
                current_queue = in_queues[queue_ctr % len(in_queues)]
                if not current_queue.empty():
                    # get the item
                    item = current_queue.get()
                    # if we do pin memory, do it now, otherwise skip this
                    if do_pin_memory:
                        if isinstance(item, dict):
                            for k in item.keys():
                                if isinstance(item[k], torch.Tensor):
                                    item[k] = item[k].pin_memory()
                    queue_ctr += 1

                    if isinstance(item, str) and item == 'end':
                        end_ctr += 1
                    if end_ctr == len(in_queues):
                        end_ctr = 0
                        queue_ctr = 0

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
        except KeyboardInterrupt:
            abort_event.set()
            raise KeyboardInterrupt


class MultiThreadedAugmenter(object):
    """ Makes your pipeline multi threaded. Yeah!
    If seeded we guarantee that batches are retunred in the same order and with the same augmentation every time this
    is run. This is realized internally by using une queue per worker and querying the queues one ofter the other.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure
        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
        num_processes (int): number of processes
        num_cached_per_queue (int): number of batches cached per process (each process has its own
        multiprocessing.Queue). We found 2 to be ideal.
        seeds (list of int): one seed for each worker. Must have len(num_processes).
        If None then seeds = range(num_processes)
        pin_memory (bool): set to True if all torch tensors in data_dict are to be pinned. Pytorch only.
        timeout (int): How long do we wait for the background workers to do stuff? If timeout seconds have passed and
        self.__get_next_item still has not gotten an item from the workers we will perform a check whether all
        background workers are still alive. If all are alive we wait, if not we set the abort flag.
        wait_time (float): set this to be lower than the time you need per iteration. Don't set this to 0,
        that will come with a performance penalty. Default is 0.02 which will be fine for 50 iterations/s
    """

    def __init__(self, data_loader, transform, num_processes, num_cached_per_queue=2, seeds=None, pin_memory=False,
                 timeout=10, wait_time=0.02):
        self.timeout = timeout
        self.pin_memory = pin_memory
        self.transform = transform
        if seeds is not None:
            assert len(seeds) == num_processes
        else:
            seeds = [None] * num_processes
        self.seeds = seeds
        self.generator = data_loader
        self.num_processes = num_processes
        self.num_cached_per_queue = num_cached_per_queue
        self._queues = []
        self._processes = []
        self._end_ctr = 0
        self._queue_ctr = 0
        self.pin_memory_thread = None
        self.pin_memory_queue = None
        self.abort_event = Event()
        self.wait_time = wait_time
        self.was_initialized = False

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __get_next_item(self):
        item = None

        while item is None:
            if self.abort_event.is_set():
                self._finish()
                raise RuntimeError("MultiThreadedAugmenter.abort_event was set, something went wrong. Maybe one of "
                                   "your workers crashed. This is not the actual error message! Look further up your "
                                   "stdout to see what caused the error. Please also check whether your RAM was full")

            if not self.pin_memory_queue.empty():
                item = self.pin_memory_queue.get()
            else:
                sleep(self.wait_time)

        return item

    def __next__(self):
        if not self.was_initialized:
            self._start()

        try:
            item = self.__get_next_item()

            while isinstance(item, str) and (item == "end"):
                self._end_ctr += 1
                if self._end_ctr == self.num_processes:
                    self._end_ctr = 0
                    self._queue_ctr = 0
                    logging.debug("MultiThreadedGenerator: finished data generation")
                    raise StopIteration

                item = self.__get_next_item()

            return item

        except KeyboardInterrupt:
            logging.error("MultiThreadedGenerator: caught exception: {}".format(sys.exc_info()))
            self.abort_event.set()
            self._finish()
            raise KeyboardInterrupt

    def _start(self):
        if not self.was_initialized:
            self._finish()
            self.abort_event.clear()

            logging.debug("starting workers")
            self._queue_ctr = 0
            self._end_ctr = 0

            if hasattr(self.generator, 'was_initialized'):
                self.generator.was_initialized = False

            with threadpool_limits(limits=1, user_api="blas"):
                for i in range(self.num_processes):
                    self._queues.append(Queue(self.num_cached_per_queue))
                    self._processes.append(Process(target=producer, args=(
                        self._queues[i], self.generator, self.transform, i, self.seeds[i], self.abort_event)))
                    self._processes[-1].daemon = True
                    self._processes[-1].start()

            if torch is not None and torch.cuda.is_available():
                gpu = torch.cuda.current_device()
            else:
                gpu = None

            # more caching = more performance. But don't cache too much or your RAM will hate you
            self.pin_memory_queue = thrQueue(max(3, self.num_cached_per_queue * self.num_processes // 2))

            self.pin_memory_thread = threading.Thread(target=results_loop, args=(
                self._queues, self.pin_memory_queue, self.abort_event, self.pin_memory, gpu, self.wait_time,
                self._processes))

            self.pin_memory_thread.daemon = True
            self.pin_memory_thread.start()

            self.was_initialized = True
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but it has already been "
                          "initialized previously")

    def _finish(self, timeout=10):
        self.abort_event.set()

        start = time()
        while self.pin_memory_thread is not None and self.pin_memory_thread.is_alive() and start + timeout > time():
            sleep(0.2)

        if len(self._processes) != 0:
            logging.debug("MultiThreadedGenerator: shutting down workers...")
            [i.terminate() for i in self._processes]

            for i, p in enumerate(self._processes):
                self._queues[i].close()
                self._queues[i].join_thread()

            self._queues = []
            self._processes = []
            self._queue = None
            self._end_ctr = 0
            self._queue_ctr = 0

            del self.pin_memory_queue
        self.was_initialized = False

    def restart(self):
        self._finish()
        self._start()

    def __del__(self):
        logging.debug("MultiThreadedGenerator: destructor was called")
        self._finish()
