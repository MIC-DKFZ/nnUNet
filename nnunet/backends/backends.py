# nnunet backend adapters by Thomas Phil
# In an ideal world these capabilities
# would be abstracted away in PyTorch
# but unfortunately xpu is not (yet)
# abstracted away in pytorch
from abc import ABC, abstractmethod

import torch

# Intel XPU imports
_intel_xpu_avail = False
try:
    import intel_extension_for_pytorch as ipex

    if ipex.xpu.is_available() and ipex.xpu.device_count() > 0:
        from intel_extension_for_pytorch.xpu.amp import autocast as xpu_autocast
        _intel_xpu_avail = True
except:
    pass

# CUDA imports
_cuda_avail = False
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    from torch.backends import cudnn
    from torch.cuda.amp import autocast as cuda_amp_autocast, GradScaler as CudaAmpGradScaler
    _cuda_avail = True



class BackendAdapter(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def set_deterministic(self, val):
        pass

    @abstractmethod
    def is_deterministic(self):
        pass

    @abstractmethod
    def set_benchmark(self, val):
        pass

    @abstractmethod
    def is_benchmark(self):
        pass
        
    @abstractmethod
    def autocast(self, *args, **kwargs):
        pass

    @abstractmethod
    def to(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_available(self):
        pass

    @abstractmethod
    def empty_cache(self):
        pass

    @abstractmethod
    def manual_seed(self, *args, **kwargs):
        pass

    @abstractmethod
    def manual_seed_all(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_device(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_gradscaler(self, *args, **kwargs):
        pass


class AutoBackend(BackendAdapter):
    def __init__(self, *args, **kwargs):
        if _cuda_avail:
            self._backend = CudaBackend(*args, **kwargs)
        elif _intel_xpu_avail:
            self._backend = IntelXPUBackend(*args, **kwargs)
        else:
            self._backend = MockBackend(*args, **kwargs)

        print(f'Using backend: {self.name()}')

    def name(self):
        return f'autobackend.{self._backend.name()}'

    def is_cuda(self):
        return is_backend_cuda(self._backend)

    def is_xpu(self):
        return is_backend_xpu(self._backend)

    def set_deterministic(self, val):
        return self._backend.set_deterministic(val)

    def is_deterministic(self):
        return self._backend.is_deterministic()

    def set_benchmark(self, val):
        return self._backend.set_benchmark(val)

    def is_benchmark(self):
        return self._backend.is_benchmark()

    def autocast(self, *args, **kwargs):
        return self._backend.autocast(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self._backend.to(*args, **kwargs)

    def is_available(self):
        return self._backend.is_available()

    def empty_cache(self):
        return self._backend.empty_cache()

    def manual_seed(self, *args, **kwargs):
        return self._backend.manual_seed(*args, **kwargs)

    def manual_seed_all(self, *args, **kwargs):
        return self._backend.manual_seed_all(*args, **kwargs)

    def set_device(self, *args, **kwargs):
        return self._backend.set_device(*args, **kwargs)

    def optimizer(self, *args, **kwargs):
        return self._backend.optimizer(*args, **kwargs)

    def get_gradscaler(self, *args, **kwargs):
        return self._backend.get_gradscaler(*args, **kwargs)


class MockContextManager():
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return


class MockBackend(BackendAdapter):
    def name(self):
        return "mock"

    def set_deterministic(self, val):
        pass

    def is_deterministic(self):
        return False

    def set_benchmark(self, val):
        pass

    def is_benchmark(self):
        return False

    def autocast(self, *args, **kwargs):
        return MockContextManager()

    def to(self, data, non_blocking=True, gpu_id=0):
        pass

    def is_available(self):
        return False

    def empty_cache(self):
        pass

    def manual_seed(self, *args, **kwargs):
        pass

    def manual_seed_all(self, *args, **kwargs):
        pass

    def set_device(self, *args, **kwargs):
        pass

    def optimizer(self, model, optimizer, *args, **kwargs):
        return model, optimizer

    def get_gradscaler(self):
        pass


class CudaBackend(BackendAdapter):
    def name(self):
        return "torch.backends.cudnn"

    def set_deterministic(self, val):
        cudnn.deterministic = val

    def is_deterministic(self):
        return cudnn.deterministic

    def set_benchmark(self, val):
        cudnn.benchmark = val

    def is_benchmark(self):
        return cudnn.benchmark

    def autocast(self, *args, **kwargs):
        return cuda_amp_autocast(*args, **kwargs)

    def to(self, data, non_blocking=True, gpu_id=0):
        if isinstance(data, list):
            data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
        else:
            data = data.cuda(gpu_id, non_blocking=non_blocking)
        return data

    def is_available(self):
        return _cuda_avail

    def empty_cache(self):
        return torch.cuda.empty_cache()

    def manual_seed(self, *args, **kwargs):
        torch.cuda.manual_seed(*args, **kwargs)

    def manual_seed_all(self, *args, **kwargs):
        torch.cuda.manual_seed_all(*args, **kwargs)

    def set_device(self, *args, **kwargs):
        torch.cuda.set_device(*args, **kwargs)

    def optimizer(self, model, optimizer, *args, **kwargs):
        return model, optimizer

    def get_gradscaler(self):
        return CudaAmpGradScaler()


class IntelXPUBackend(BackendAdapter):
    def name(self):
        return "intel_extension_for_pytorch.xpu"

    def set_deterministic(self, val):
        pass

    def is_deterministic(self):
        return False

    def set_benchmark(self, val):
        pass

    def is_benchmark(self):
        return False

    def autocast(self, dtype=None, *args, **kwargs):
        if dtype == torch.float16:
            dtype = torch.bfloat16

        # Intel ARC only supports 16 and bits at the time of writing this
        # at some point we should enable some autodetect for compatibility
        supported_dtypes = [torch.bfloat16]  # last one should be highest order

        if dtype is None:
            dtype = supported_dtypes[-1]
        elif dtype not in supported_dtypes:
            old = dtype
            dtype = supported_dtypes[-1]  # last one should be highest order
            print(f'WARN: {self.name()} autocast requested unsupported dtype {old} - autocasting to {dtype} instead')

        return xpu_autocast(dtype=dtype, enabled=True, cache_enabled=False, *args, **kwargs) 

    def to(self, obj, non_blocking=True, *args, **kwargs):
        if isinstance(obj, list):
            obj = [i.to('xpu') for i in obj]
        else:
            obj = obj.to('xpu')
        return obj

    def is_available(self):
        return _intel_xpu_avail

    def empty_cache(self):
        return ipex.xpu.empty_cache()

    def manual_seed(self, *args, **kwargs):
        return ipex.xpu.manual_seed(*args, **kwargs)
        
    def manual_seed_all(self, *args, **kwargs):
        return ipex.xpu.manual_seed_all(*args, **kwargs)

    def set_device(self, *args, **kwargs):
        return ipex.xpu.set_device(*args, **kwargs)

    def optimizer(self, model, optimizer, dtype=torch.bfloat16, *args, **kwargs):
        return ipex.optimize(model, optimizer=optimizer, dtype=dtype)

    def get_gradscaler(self, *args, **kwargs):
        return None

def is_backend_cuda(backend):
    if isinstance(backend, AutoBackend):
        return backend.is_cuda()

    return isinstance(backend, [CudaBackend])

def is_backend_xpu(backend):
    if isinstance(backend, AutoBackend):
        return backend.is_xpu()

    return isinstance(backend, [IntelXPUBackend])
