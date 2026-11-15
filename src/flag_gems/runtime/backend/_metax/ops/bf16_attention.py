# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""C500 BF16 GQA decode with fixed page arithmetic and shared K/V storage."""

import ctypes
import fcntl
import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("VLLM_FL_METAX_NATIVE_DECODE", "1") == "1"
_FLAGS = (
    "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
    "--offload-arch=xcore1000", "-DUSE_MACA", "-DNV_ARCH_A100",
    "-D__FAST_HALF_CVT__", "-D__MERGE_LDS_B64",
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
    "-fno-strict-aliasing", "-Xclang", "-menable-no-nans",
    "-mllvm", "-metaxgpu-inlinescope=50",
    "-mllvm", "-metaxgpu-disable-bsm-offset=0",
    "-mllvm", "-metaxgpu-enable-promote-kernel-arguments",
    "-mllvm", "-metaxgpu-const-KArgs-LDU=1",
    "-mllvm", "-metaxgpu-enable-gvn-loadclobber=false",
    "-mllvm", "-metaxgpu-enable-unorder-dispatch",
    "-mllvm", "-metaxgpu-disable-early-vector-combine=true",
    "-mllvm", "-metaxgpu-preisel-simplifycfg=true",
    "-mllvm", "-metaxgpu-igroup",
    "-mllvm", "-metaxgpu-igroup-strategy=1",
)


@functools.lru_cache(maxsize=None)
def _is_c500(device):
    return "C500" in torch.cuda.get_device_name(device)


@functools.lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_name("bf16_attention_native")
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    compiler = maca / "tools/cu-bridge/bin/cucc"
    try:
        digest = hashlib.sha256()
        for path in sorted(source.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
        digest.update(repr(_FLAGS).encode())
        digest.update(str(compiler.resolve()).encode())
        digest.update(str(compiler.stat().st_mtime_ns).encode())
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache = cache / "flag_gems/metax_attention" / digest.hexdigest()[:24]
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / "decode.so"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not library.exists():
                temporary = cache / f"decode.{os.getpid()}.so"
                command = [str(compiler), *_FLAGS, "-I" + str(source),
                           "-I" + str(source / "include"),
                           "-I" + str(maca / "include"),
                           "-I" + str(maca / "tools/cu-bridge/include"),
                           str(source / "decode.cu"), "-o", str(temporary)]
                with (cache / "build.log").open("w") as build_log:
                    subprocess.run(command, stdout=build_log,
                                   stderr=subprocess.STDOUT, check=True)
                temporary.replace(library)
        handle = ctypes.CDLL(str(library))
        function = handle.flag_gems_c500_decode
        function.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        function.restype = ctypes.c_int
        logger.info("Loaded C500 BF16 GQA decode kernel: %s", library)
        return handle, function
    except (OSError, subprocess.SubprocessError):
        logger.warning("C500 native BF16 decode unavailable; using vendor FlashAttention",
                       exc_info=True)
        return None


def prepare_bf16_decode(device):
    """Compile and register before CUDA graph capture, without running a kernel."""
    return _ENABLED and _is_c500(device) and _kernel() is not None


def try_bf16_decode(q, key_cache, value_cache, output, block_table, lengths, scale):
    """Return False for unsupported inputs, retaining the normal vendor dispatch."""
    if not _ENABLED:
        return False
    batch = q.shape[0]
    if (q.ndim != 4 or q.shape[1:] != (1, 16, 128) or not 33 <= batch <= 64
            or q.dtype != torch.bfloat16 or q.device.type != "cuda"
            or q.stride(-1) != 1 or q.stride(2) != 128 or q.stride(0) < 2048
            or output.shape != q.shape or not output.is_contiguous()
            or key_cache.ndim != 4 or key_cache.shape != value_cache.shape
            or key_cache.shape[1:] != (16, 2, 128)
            or key_cache.stride() != (8192, 128, 4096, 1)
            or value_cache.stride() != key_cache.stride()
            or any(t.dtype != q.dtype or t.device != q.device
                   for t in (key_cache, value_cache, output))
            or block_table is None or block_table.ndim != 2
            or block_table.shape[0] < batch or block_table.shape[1] == 0
            or block_table.stride(1) != 1 or block_table.stride(0) < block_table.shape[1]
            or block_table.dtype != torch.int32 or block_table.device != q.device
            or lengths is None or lengths.shape != (batch,)
            or lengths.dtype != torch.int32 or lengths.device != q.device
            or not lengths.is_contiguous() or abs(scale - 128 ** -0.5) > 1e-8
            or not _is_c500(q.device.index)):
        return False
    kernel = _kernel()
    if kernel is None:
        return False
    partials = torch.empty((6, batch, 2, 8, 128), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty((6, batch, 2, 8), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, 2, 8), device=q.device, dtype=torch.float32)
    _, function = kernel
    with torch.cuda.device(q.device):
        status = function(
            q.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
            output.data_ptr(), block_table.data_ptr(), lengths.data_ptr(),
            partials.data_ptr(), partial_lse.data_ptr(), lse.data_ptr(),
            batch, q.stride(0), block_table.stride(0),
            torch.cuda.current_stream().cuda_stream,
        )
    if status:
        raise RuntimeError(f"C500 BF16 decode launch failed with status {status}")
    return True
