# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""C500 BF16 prefill GEMMs with coalesced output and optional rounded SwiGLU.

The native kernel reads the existing N128/K64 packed weight cache. It uses
four BSM staging buffers and reuses their shared memory for the output
transpose after the MMA pipeline has drained. Accumulation stays FP32.
"""

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

_FLAGS = (
    "-O3", "-DENABLE_BF16", "--use_fast_math", "-std=c++17",
    "-shared", "-Xcompiler", "-fPIC", "--offload-arch=xcore1000",
    "-mllvm", "-metaxgpu-direct-address=disshared",
    "-mllvm", "-metaxgpu-force-global-saddr=1",
    "-mllvm", "-metaxgpu-sched-regpressure=true",
    "-mllvm", "-metaxgpu-sinkload=false",
    "-mllvm", "-metaxgpu-vectorize-slp=true",
    "-mllvm", "-metaxgpu-igroup",
    "-mllvm", "-metaxgpu-igroup-strategy=1",
    "-mllvm", "-metaxgpu-aggressive-4g-addr-opt=true",
    "-mllvm", "-metaxgpu-shl-add-combine=false",
    "-mllvm", "-misched-postra=true",
    "-mllvm", "-enable-post-misched=true",
)


@functools.lru_cache(maxsize=None)
def _is_c500(device_index):
    return "C500" in torch.cuda.get_device_name(device_index)


def _load_library(source_name, flags):
    source = Path(__file__).with_name("bsm_gate_native")
    maca = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    compiler = maca / "tools/cu-bridge/bin/cucc"
    digest = hashlib.sha256()
    for path in sorted(source.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(repr(flags).encode())
    digest.update(str(compiler.resolve()).encode())
    digest.update(str(compiler.stat().st_mtime_ns).encode())
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    cache = cache / "flag_gems/metax_bsm" / digest.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"{source_name}.so"
    with (cache / f"{source_name}.build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = cache / f"{source_name}.{os.getpid()}.so"
            command = [str(compiler), *flags,
                       "-I" + str(maca / "include"),
                       "-I" + str(maca / "tools/cu-bridge/include"),
                       "-I" + str(source), str(source / f"{source_name}.cu"),
                       "-o", str(temporary)]
            with (cache / f"{source_name}.build.log").open("w") as build_log:
                subprocess.run(command, stdout=build_log,
                               stderr=subprocess.STDOUT, check=True)
            temporary.replace(library)
    logger.info("Loaded C500 BF16 BSM %s kernels: %s", source_name, library)
    return ctypes.CDLL(str(library))


@functools.lru_cache(maxsize=1)
def _kernel():
    try:
        handle = _load_library("gate", _FLAGS)
        function = handle.run_packed_swiglu
        function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        linear = handle.run_packed_linear
        linear.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        linear.restype = ctypes.c_int
        return handle, function, linear
    except (OSError, subprocess.SubprocessError):
        logger.warning("C500 BSM prefill kernel unavailable; using Triton", exc_info=True)
        return None


@functools.lru_cache(maxsize=1)
def _tail_kernel():
    # Dynamic split addresses use the ordinary 64-bit global-address path.
    unsupported = {"-metaxgpu-force-global-saddr=1",
                   "-metaxgpu-aggressive-4g-addr-opt=true"}
    flags = []
    i = 0
    while i < len(_FLAGS):
        if _FLAGS[i] == "-mllvm" and _FLAGS[i + 1] in unsupported:
            i += 2
        else:
            flags.append(_FLAGS[i])
            i += 1
    try:
        handle = _load_library("tail", tuple(flags))
        function = handle.run_bsm_tail
        function.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
                             + [ctypes.c_void_p])
        function.restype = ctypes.c_int
        return handle, function
    except (OSError, subprocess.SubprocessError):
        logger.warning("C500 BSM tail kernel unavailable; using regular GEMM", exc_info=True)
        return None


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _try_tail_linear(a, packed, output):
    if (os.environ.get("FLAGGEMS_METAX_BSM_TAIL", "1") != "1"
            or a.shape[1] != 6144 or output.shape[1] != 2048
            or not 1921 <= a.shape[0] <= 2048 or _sm_count(a.device.index) != 104):
        return False
    kernel = _tail_kernel()
    if kernel is None:
        return False
    partials = torch.empty((2, a.shape[0], 2048), device=a.device, dtype=torch.float32)
    _, function = kernel
    with torch.cuda.device(a.device):
        status = function(a.data_ptr(), packed.data_ptr(), output.data_ptr(),
                          partials.data_ptr(), a.shape[0], 2048, 6144, 2,
                          _sm_count(a.device.index), torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"C500 BSM tail GEMM launch failed with status {status}")
    return True


def packed_gate_into(a, packed, output):
    """Return False when this device/toolchain needs the existing Triton path."""
    if not _is_c500(a.device.index):
        return False
    kernel = _kernel()
    if kernel is None:
        return False
    _, function, _ = kernel
    with torch.cuda.device(a.device):
        status = function(a.data_ptr(), packed.data_ptr(), output.data_ptr(),
                          a.shape[0], torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"C500 BSM prefill launch failed with status {status}")
    return True


def packed_linear_into(a, packed, output):
    """Launch a BF16 linear GEMM using the existing N128/K64 packed weights."""
    if not _is_c500(a.device.index):
        return False
    if _try_tail_linear(a, packed, output):
        return True
    kernel = _kernel()
    if kernel is None:
        return False
    _, _, function = kernel
    with torch.cuda.device(a.device):
        status = function(a.data_ptr(), packed.data_ptr(), output.data_ptr(),
                          a.shape[0], output.shape[1], a.shape[1],
                          torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f"C500 BSM prefill linear launch failed with status {status}")
    return True
