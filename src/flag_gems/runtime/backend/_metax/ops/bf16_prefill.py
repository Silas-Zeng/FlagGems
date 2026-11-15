# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""C500 long prefill using the regular attention pipeline and FP32 split-KV."""
import ctypes
import fcntl
import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl

from .bf16_attention import _FLAGS, _is_c500

logger = logging.getLogger(__name__)
_selection_logged = False
_ENABLED = os.environ.get("VLLM_FL_METAX_NATIVE_PREFILL", "1") == "1"


@functools.lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_name("bf16_prefill_native")
    shared = Path(__file__).with_name("bf16_attention_native")
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    compiler = maca / "tools/cu-bridge/bin/cucc"
    try:
        digest = hashlib.sha256()
        for root in (source, shared):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    digest.update(str(path.relative_to(root.parent)).encode())
                    digest.update(path.read_bytes())
        digest.update(repr(_FLAGS).encode())
        digest.update(str(compiler.resolve()).encode())
        digest.update(str(compiler.stat().st_mtime_ns).encode())
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache = cache / "flag_gems/metax_prefill" / digest.hexdigest()[:24]
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / "prefill.so"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.exists():
                temporary = cache / f"prefill.{os.getpid()}.so"
                command = [str(compiler), *_FLAGS, "-I" + str(source),
                           "-I" + str(shared), "-I" + str(shared / "include"),
                           "-I" + str(maca / "include"),
                           "-I" + str(maca / "tools/cu-bridge/include"),
                           str(source / "prefill.cu"), "-o", str(temporary)]
                with (cache / "build.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                temporary.replace(library)
        handle = ctypes.CDLL(str(library))
        function = handle.flag_gems_c500_prefill
        function.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        function.restype = ctypes.c_int
        logger.info("Loaded C500 FP32 split-KV prefill: %s", library)
        return handle, function
    except (OSError, subprocess.SubprocessError):
        logger.warning("C500 native prefill unavailable; retaining vendor path", exc_info=True)
        return None


@triton.jit(do_not_specialize=["M"])
def _merge_parts(P, L, O, M, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < M * 2048
    row, head = i // 2048, (i // 128) % 16
    maximum = tl.full((B,), -float("inf"), tl.float32)
    for s in range(S):
        lse = tl.load(L + s * M * 16 + head * M + row, valid, -float("inf"))
        maximum = tl.maximum(maximum, lse)
    numerator = tl.full((B,), 0, tl.float32)
    denominator = tl.full((B,), 0, tl.float32)
    for s in range(S):
        lse = tl.load(L + s * M * 16 + head * M + row, valid, -float("inf"))
        weight = tl.exp(lse - maximum)
        part = tl.load(P + s * M * 2048 + i, valid, 0)
        # A partition can be wholly beyond this query's causal boundary.
        # Ignore its undefined normalized output when its LSE weight is zero.
        numerator += tl.where(weight > 0, weight * part, 0.)
        denominator += weight
    tl.store(O + i, numerator / denominator, valid)


def try_bf16_prefill(q, key, value, output, scale):
    """Dense single-request input with exact CPU-known sequence lengths."""
    global _selection_logged
    if not _ENABLED:
        return False
    if (q.ndim != 3 or q.shape[1:] != (16, 128)
            or q.device.type != "cuda" or q.dtype != torch.bfloat16
            or not 1024 <= q.shape[0] <= 2048
            or key.ndim != 3 or key.shape[1:] != (2, 128)
            or not 8192 <= key.shape[0] <= 16384
            or key.shape != value.shape
            or q.stride(-1) != 1 or q.stride(1) != 128 or q.stride(0) < 2048
            or key.stride() != (256, 128, 1) or value.stride() != key.stride()
            or output.shape != q.shape or not output.is_contiguous()
            or any(x.dtype != q.dtype or x.device != q.device for x in (key, value, output))
            or any(x.requires_grad for x in (q, key, value, output))
            or abs(scale - 128 ** -.5) > 1e-8 or not _is_c500(q.device.index)):
        return False
    kernel = _kernel()
    if kernel is None:
        return False
    m, n = q.shape[0], key.shape[0]
    splits = 4 if m <= 1408 and n >= 12288 else 2
    if not _selection_logged and os.environ.get("VLLM_FL_METAX_DECODE_LAYOUT_LOG", "0") == "1":
        logger.info("C500 native prefill selected: query=%s key=%s splits=%s FP32_partials=True", m, n, splits)
        _selection_logged = True
    partials = torch.empty((splits, m, 16, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((splits, 16, m), device=q.device, dtype=torch.float32)
    _, function = kernel
    with torch.cuda.device(q.device):
        status = function(q.data_ptr(), key.data_ptr(), value.data_ptr(),
                          partials.data_ptr(), None, None, lse.data_ptr(),
                          m, n, q.stride(0), key.stride(0), key.stride(1), splits,
                          torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"C500 prefill launch failed with status {status}")
        _merge_parts[(triton.cdiv(m * 2048, 1024),)](
            partials, lse, output, m, splits, 1024, num_warps=4,
        )
    return True
