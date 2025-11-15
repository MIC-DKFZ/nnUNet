# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
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

import threading
from queue import Queue as ThreadQueue

class ThreadedIterator:
    def __init__(self, data_loader, num_threads=4):
        self.data_loader = data_loader
        self.queue = ThreadQueue(maxsize=10)
        self.threads = []

        for i in range(num_threads):
            t = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            t.start()
            self.threads.append(t)

    def _worker(self, thread_id):
        while True:
            # No serialization - threads share memory!
            item = next(self.data_loader)
            self.queue.put(item)

    def __next__(self):
        return self.queue.get()