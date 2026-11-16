"""
Triton kernels for nnUNetPredictorStreaming. Both reproduce the predictor's torch path bit for bit, for an fp16 and an
fp32 ring; the predictor uses them on CUDA when Triton imports (no Windows wheels) and kernels_selftest passes for the
ring dtype (compute capability >= 7.0, both kernels bit-identical, no exception), and keeps the torch path otherwise.
Triton launches a kernel on the calling thread's current CUDA device, so both wrappers make the device of their tensors
current for the launch: a predictor on cuda:1 works while cuda:0 is current.
Tests: nnunetv2/tests/test_predict_streaming_kernels.py.

accumulate_: ring_slice += pred * gaussian in one pass instead of two, with torch's roundings: round16(pred * g)
    (pred and g are fp16), then ring + . rounded to the ring dtype -> identical to `pred.mul_(g); ring_slice += pred`.
emit_argmax: one target plane straight from the ring, the torch emit's arithmetic operation for operation:
    normalisation in the ring dtype (IEEE fp32 divide, rounded to fp16 for an fp16 ring), bilinear per source slice
    with ATen's coordinate rule and the fma contraction nvcc applied when it compiled ATen's upsample_bilinear2d, slab
    blend with float32(1 - w), and the first-maximum argmax as a running max over channels -> uint8/int16 labels.
    Nothing C-sized exists at output resolution (the torch path holds ~3 (C, target cross-section) fp32 planes). The
    contraction pattern is a property of the torch build, hence kernels_selftest.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from torch.cuda import device as cuda_device


@triton.jit
def _accumulate_kernel(ring_ptr, pred_ptr, g_ptr, rs_c, rs_z, rs_y, n_zy, P1: tl.constexpr, P2: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)         # one (channel, z, y) row of the patch; int64: rings exceed 2^31
    c = row // n_zy
    zy = row % n_zy
    z = zy // P1
    y = zy % P1
    x = tl.arange(0, BLOCK)
    m = x < P2
    p = tl.load(pred_ptr + row * P2 + x, mask=m).to(tl.float32)
    gv = tl.load(g_ptr + zy * P2 + x, mask=m).to(tl.float32)
    prod = (p * gv).to(tl.float16).to(tl.float32)
    off = c * rs_c + z * rs_z + y * rs_y + x
    r = tl.load(ring_ptr + off, mask=m).to(tl.float32)
    tl.store(ring_ptr + off, (r + prod).to(ring_ptr.dtype.element_ty), mask=m)     # fp16 ring: round; fp32: as is


def accumulate_(ring_slice: torch.Tensor, pred: torch.Tensor, gaussian: torch.Tensor):
    """ring_slice (C, a, b, c) fp16 or fp32 view with unit last stride; pred (C, a, b, c) and gaussian (a, b, c) fp16."""
    C, P0, P1, P2 = pred.shape
    pred = pred.contiguous()
    gaussian = gaussian.contiguous()
    assert ring_slice.stride(3) == 1 and ring_slice.dtype in (torch.float16, torch.float32)
    assert pred.dtype == gaussian.dtype == torch.float16
    with cuda_device(ring_slice.device):             # Triton launches on the current device and its stream
        _accumulate_kernel[(C * P0 * P1,)](ring_slice, pred, gaussian, ring_slice.stride(0), ring_slice.stride(1),
                                           ring_slice.stride(2), P0 * P1, P1=P1, P2=P2,
                                           BLOCK=triton.next_power_of_2(P2), enable_fp_fusion=False)


def bilinear_tables(src_len: int, dst_len: int, device, select: torch.Tensor = None):
    """Per output index: (i0, i1, lambda1) exactly as ATen's upsample_bilinear2d (align_corners=False) computes them:
    scale = float32(src) / float32(dst), source coordinate = fma(scale, dst + 0.5, -0.5) (nvcc contracts that line;
    one rounding), clamped at 0. With `select` (separate-z axis in the cross-section: nearest source index per output)
    lambda1 = 0."""
    if select is not None:
        idx = select.to(device=device, dtype=torch.int32)
        return idx, idx, torch.zeros(len(idx), dtype=torch.float32, device=device)
    scale = np.float32(np.float32(src_len) / np.float32(dst_len))
    d = np.arange(dst_len, dtype=np.float32)
    src = (np.float64(scale) * (d + np.float32(0.5)).astype(np.float64) - 0.5).astype(np.float32)   # = fma
    src = np.maximum(src, np.float32(0))
    i0 = src.astype(np.int32)
    i1 = np.where(i0 < src_len - 1, i0 + 1, i0).astype(np.int32)
    lam1 = (src - i0.astype(np.float32)).astype(np.float32)
    return tuple(torch.from_numpy(x).to(device) for x in (i0, i1, lam1))


@triton.jit
def _norm(logit, npred, RING_FP16: tl.constexpr):
    """torch's division in the ring dtype: IEEE fp32 divide, rounded to fp16 for an fp16 ring."""
    q = tl.math.div_rn(logit, npred)
    if RING_FP16:
        q = q.to(tl.float16).to(tl.float32)
    return q


@triton.jit
def _bilinear_norm(lp, np_, base, oaa, oab, oba, obb, naa, nab, nba, nbb, m, h0, h1, w0, w1, RING_FP16: tl.constexpr):
    """One channel of one source slice: normalisation as torch's division in the ring dtype, then ATen's bilinear
    expression with nvcc's contraction: fma(h0, fma(w0, a, w1*b), h1*fma(w0, c, w1*d))."""
    a = _norm(tl.load(lp + base + oaa, mask=m, other=0.).to(tl.float32), naa, RING_FP16)
    b = _norm(tl.load(lp + base + oab, mask=m, other=0.).to(tl.float32), nab, RING_FP16)
    c = _norm(tl.load(lp + base + oba, mask=m, other=0.).to(tl.float32), nba, RING_FP16)
    d = _norm(tl.load(lp + base + obb, mask=m, other=0.).to(tl.float32), nbb, RING_FP16)
    return tl.fma(h0, tl.fma(w0, a, w1 * b), h1 * tl.fma(w0, c, w1 * d))


@triton.jit
def _emit_kernel(l0_ptr, l1_ptr, n0_ptr, n1_ptr, out_ptr, best_ptr, a0_ptr, a1_ptr, al_ptr, b0_ptr, b1_ptr, bl_ptr,
                 w, wc, C, ls_c, ls_y, ns_y, T1, BLOCK: tl.constexpr, BLEND: tl.constexpr, STORE_BEST: tl.constexpr,
                 RING_FP16: tl.constexpr):
    t0 = tl.program_id(0)
    t1 = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = t1 < T1
    ya = tl.load(a0_ptr + t0)
    yb = tl.load(a1_ptr + t0)
    h1 = tl.load(al_ptr + t0)
    h0 = 1.0 - h1
    xa = tl.load(b0_ptr + t1, mask=m, other=0)
    xb = tl.load(b1_ptr + t1, mask=m, other=0)
    w1 = tl.load(bl_ptr + t1, mask=m, other=0.)
    w0 = 1.0 - w1
    oaa = ya * ls_y + xa
    oab = ya * ls_y + xb
    oba = yb * ls_y + xa
    obb = yb * ls_y + xb
    n0aa = tl.load(n0_ptr + ya * ns_y + xa, mask=m, other=1.).to(tl.float32)
    n0ab = tl.load(n0_ptr + ya * ns_y + xb, mask=m, other=1.).to(tl.float32)
    n0ba = tl.load(n0_ptr + yb * ns_y + xa, mask=m, other=1.).to(tl.float32)
    n0bb = tl.load(n0_ptr + yb * ns_y + xb, mask=m, other=1.).to(tl.float32)
    n1aa = tl.load(n1_ptr + ya * ns_y + xa, mask=m, other=1.).to(tl.float32)
    n1ab = tl.load(n1_ptr + ya * ns_y + xb, mask=m, other=1.).to(tl.float32)
    n1ba = tl.load(n1_ptr + yb * ns_y + xa, mask=m, other=1.).to(tl.float32)
    n1bb = tl.load(n1_ptr + yb * ns_y + xb, mask=m, other=1.).to(tl.float32)
    best = tl.full([BLOCK], float('-inf'), tl.float32)
    arg = tl.zeros([BLOCK], tl.int32)
    for c in range(0, C):
        base = c.to(tl.int64) * ls_c            # ring channel stride * C can exceed 2^31 (126 classes, large images)
        v = _bilinear_norm(l0_ptr, n0_ptr, base, oaa, oab, oba, obb, n0aa, n0ab, n0ba, n0bb, m, h0, h1, w0, w1,
                           RING_FP16)
        if BLEND:                               # torch: (1 - w) * sl0 + w * sl1, three separately rounded ops
            v1 = _bilinear_norm(l1_ptr, n1_ptr, base, oaa, oab, oba, obb, n1aa, n1ab, n1ba, n1bb, m, h0, h1, w0, w1,
                                RING_FP16)
            v = wc * v + w * v1
        upd = v > best                          # strict: first maximum wins, as torch.argmax
        best = tl.where(upd, v, best)
        arg = tl.where(upd, c, arg)
    tl.store(out_ptr + t0 * T1 + t1, arg.to(out_ptr.dtype.element_ty), mask=m)
    if STORE_BEST:                              # self-test only: the winning value, for a bitwise comparison
        tl.store(best_ptr + t0 * T1 + t1, best, mask=m)


def emit_argmax(l0: torch.Tensor, l1: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor, w: float, blend: bool,
                tab0, tab1, out: torch.Tensor, block: int = 128, best: torch.Tensor = None):
    """One target plane, bit-identical to the torch emit of nnUNetPredictorStreaming (this torch build; see
    selftest). l0/l1: (C, s0, s1) ring views (unit last stride, same strides); n0/n1: (s0, s1); all four fp16 or
    all four fp32 (the ring dtype); out (T0, T1) uint8/int16. blend=False: the plane copies source slice 0 (i0 == i1)."""
    C = l0.shape[0]
    T0, T1 = out.shape
    assert l0.stride() == l1.stride() and l0.stride(2) == 1 and n0.stride() == n1.stride() and n0.stride(1) == 1
    assert l0.dtype == l1.dtype == n0.dtype == n1.dtype and l0.dtype in (torch.float16, torch.float32)
    assert out.is_contiguous()
    grid = (T0, triton.cdiv(T1, block))
    # torch multiplies a float32 tensor by the python double (1.0 - w) rounded to float32, not by 1 - float32(w)
    with cuda_device(out.device):                    # Triton launches on the current device and its stream
        _emit_kernel[grid](l0, l1, n0, n1, out, out if best is None else best, *tab0, *tab1, float(w), float(1.0 - w),
                           C, l0.stride(0), l0.stride(1), n0.stride(0), T1, BLOCK=block, BLEND=bool(blend),
                           STORE_BEST=best is not None, RING_FP16=l0.dtype == torch.float16, enable_fp_fusion=False)


_SELFTEST = {}


def _selftest_cases(device, ring_dtype):
    import torch.nn.functional as F
    g = torch.Generator(device=device).manual_seed(0)
    # accumulate: bitwise against `pred.mul_(g); ring_slice += pred`, permuted pred included
    ring = (torch.randn(3, 20, 24, 28, device=device, generator=g) * 3).to(ring_dtype)
    pred = (torch.randn(3, 10, 12, 14, device=device, generator=g) * 5).half().permute(0, 2, 1, 3)
    gauss = (torch.rand(12, 10, 14, device=device, generator=g) * 10).half()
    sl = (slice(None), slice(2, 14), slice(3, 13), slice(4, 18))
    ref = ring.clone()
    ref[sl] += pred * gauss
    accumulate_(ring[sl], pred, gauss)
    if not torch.equal(ring, ref):
        return False
    # emit: winning values bitwise (labels alone hide last-bit drift; they rarely flip on random data)
    for (C, S0, S1, T0, T1, w) in ((7, 61, 47, 97, 31, 0.6180339887498949), (5, 90, 64, 41, 128, 0.25)):
        n0 = (torch.rand(S0, S1, device=device, generator=g) * 30 + 1).to(ring_dtype)
        n1 = (torch.rand(S0, S1, device=device, generator=g) * 30 + 1).to(ring_dtype)
        l0 = (torch.randn(C, S0, S1, device=device, generator=g) * 400).to(ring_dtype)
        l1 = (torch.randn(C, S0, S1, device=device, generator=g) * 400).to(ring_dtype)
        b0 = F.interpolate((l0 / n0).float()[None], size=(T0, T1), mode='bilinear', align_corners=False)[0]
        b1 = F.interpolate((l1 / n1).float()[None], size=(T0, T1), mode='bilinear', align_corners=False)[0]
        ref_val, ref_arg = ((1.0 - w) * b0 + w * b1).max(0)
        out = torch.empty((T0, T1), dtype=torch.uint8, device=device)
        best = torch.empty((T0, T1), dtype=torch.float32, device=device)
        emit_argmax(l0, l1, n0, n1, w, True, bilinear_tables(S0, T0, device), bilinear_tables(S1, T1, device), out,
                    best=best)
        if not (torch.equal(out, ref_arg.to(torch.uint8)) and torch.equal(best, ref_val)):
            return False
    return True


def kernels_selftest(device, ring_dtype: torch.dtype = torch.float16) -> bool:
    """
    Once per process, device and ring dtype (fp16 / fp32): may the predictor use these kernels? Needs compute
    capability >= 7.0 (Triton's minimum; older GPUs import Triton fine but fail to compile) and both kernels
    reproducing the torch path bit for bit. The emit's fma pattern is a property of how ATen was compiled, so a
    different torch build may differ; any failure or exception (compile error, unsupported GPU) means the predictor
    keeps the torch path.
    """
    if ring_dtype not in (torch.float16, torch.float32):
        return False
    key = (str(device), ring_dtype)
    if key not in _SELFTEST:
        try:
            ok = torch.cuda.get_device_capability(device) >= (7, 0) and _selftest_cases(device, ring_dtype)
        except Exception as e:
            print(f'[streaming] Triton kernels unavailable on {device} ({type(e).__name__}: {e}); using the torch path')
            ok = False
        _SELFTEST[key] = ok
    return _SELFTEST[key]
