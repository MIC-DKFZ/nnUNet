import threading
from queue import Queue as ThreadQueue
import torch
import numpy as np
import time



def pin_memory_of_all_eligible_items_in_dict(result_dict, pin_memory):
    result_dict['data'] = torch.from_numpy(result_dict['data'])
    result_dict['target'] = torch.from_numpy(result_dict['target']).to(torch.int16)
    if pin_memory:
        result_dict['data'] = result_dict['data'].pin_memory()
        result_dict['target'] = result_dict['target'].pin_memory()
    return result_dict



class ThreadedGPUAugmenter:
    """
    Threaded data loader with built-in GPU augmentation.

    Workers load data in parallel (shared memory via threads), then
    GPU augmentation is applied on the main thread with optional CUDA streams.

    Args:
        data_loader: Base data loader (generates batches)
        gpu_transform: Transform to apply on GPU (or None)
        num_threads: Number of loading threads
        queue_size: Size of the prefetch queue
    """

    def __init__(self, data_loader, gpu_transforms=None, num_threads=4, queue_size=10):
        self.data_loader = data_loader
        self.gpu_transforms = gpu_transforms
        self.device = torch.device('cuda')

        # Queue for loaded data (before GPU augmentation)
        self.load_queue = ThreadQueue(maxsize=queue_size)


        pinned_queue_size = 2
        self.data_queue = ThreadQueue(maxsize=pinned_queue_size)

        # Thread management
        self.threads = []
        self.stop_event = threading.Event()

        # Start loading threads
        for i in range(num_threads):
            t = threading.Thread(
                target=self._loading_worker,
                args=(i,),
                daemon=True
            )
            t.start()
            self.threads.append(t)

        self.pin_thread = threading.Thread(target=self._pin_worker, daemon=True)
        self.pin_thread.start()

    def _loading_worker(self, thread_id):
        """Worker thread that loads data (CPU side)"""
        try:
            while not self.stop_event.is_set():
                try:
                    item = next(self.data_loader)
                    self.load_queue.put(item)
                except StopIteration:
                    break
        except Exception as e:
            print(f"Loading worker {thread_id} error: {e}")
            import traceback
            traceback.print_exc()

    def _pin_worker(self):
        """Convert to torch and pin memory"""
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.load_queue.get(timeout=0.1)
                    # Pin memory
                    item = pin_memory_of_all_eligible_items_in_dict(item, True)
                    self.data_queue.put(item)
                except:
                    if self.stop_event.is_set():
                        break
        except Exception as e:
            print(f"Pin worker error: {e}")

    @torch.no_grad()
    def _apply_gpu_transform(self, batch):
        """Apply GPU transformation to a batch"""
        t1 = time.time()
        data_all = batch['data'].cuda(non_blocking=True)
        seg_all = batch['target'].cuda(non_blocking=True)

        t2 = time.time()
        print(f"{t2-t1} s for device transfer")
        t1 = time.time()
        images = []
        segs = []
        batch_size = len(data_all)
        for b in range(batch_size):
            tmp = self.gpu_transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
            images.append(tmp['image'])
            segs.append(tmp['segmentation'])
        data_all = torch.stack(images)
        if isinstance(segs[0], list):
            seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
        else:
            seg_all = torch.stack(segs).to(torch.long)
        del segs, images

        t2 = time.time()
        print(f"{t2-t1} s for gpu augmentation!")
        return {'data': data_all, 'target': seg_all}

    def __next__(self):
        t1 = time.time()
        item = self.data_queue.get()
        t2 = time.time()
        print(f"{t2-t1} s to retrieve data from queue")
        item = self._apply_gpu_transform(item)
        return item

    def __iter__(self):
        return self

    def __del__(self):
        """Cleanup on deletion"""
        self.stop_event.set()

