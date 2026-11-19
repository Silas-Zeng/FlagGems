# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch


@pytest.fixture(scope="module")
def weight():
    torch.manual_seed(20261103)
    return torch.randn(130560, 2048, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "shape",
    [(2048,), (8, 2048), (16, 2048), (32, 2048), (33, 2048), (64, 2048), (2, 4, 2048)],
)
def test_wide_linear_vendor_and_graph(shape, weight):
    import flag_gems

    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    fn = flag_gems.linear
    assert fn.__module__.endswith("_metax.ops.linear")
    for _ in range(3):
        result = fn(a, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn(a, weight)
    for _ in range(2):
        a.normal_()
        graph.replay()
        reference = torch.nn.functional.linear(a, weight)
        torch.testing.assert_close(result, reference, atol=0.1, rtol=0.03)


@pytest.mark.parametrize("kind", ["bias", "fp32", "strided", "large_batch"])
def test_linear_fallback(kind):
    fn = importlib.import_module("flag_gems.runtime.backend._metax.ops.linear").linear
    dtype = torch.float32 if kind == "fp32" else torch.bfloat16
    m = 65 if kind == "large_batch" else 8
    a = torch.randn(m, 2048, device="cuda", dtype=dtype)
    if kind == "strided":
        a = torch.randn(m, 4096, device="cuda", dtype=dtype)[:, ::2]
    w = torch.randn(32768, 2048, device="cuda", dtype=dtype)
    bias = torch.randn(32768, device="cuda", dtype=dtype) if kind == "bias" else None
    torch.testing.assert_close(
        fn(a, w, bias), torch.nn.functional.linear(a, w, bias), atol=0.1, rtol=0.03
    )


@pytest.mark.parametrize("m,n", [(1, 32769), (8, 130560), (33, 130560), (64, 130560)])
def test_row_division_vendor_and_graph(m, n):
    import flag_gems

    fn = flag_gems.true_divide_
    assert fn.__module__.endswith("_metax.ops.div")
    source = torch.randn(m, n, device="cuda")
    a = source.clone()
    b = torch.rand(m, 1, device="cuda") + 0.01
    for _ in range(3):
        fn(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn(a, b)
    for temp in [0.5, 1.0, 0.73]:
        a.copy_(source)
        b.fill_(temp)
        expected = source / b
        graph.replay()
        assert result.data_ptr() == a.data_ptr()
        torch.testing.assert_close(a, expected, atol=1e-6, rtol=2e-6)


def test_row_division_special_values():
    fn = importlib.import_module("flag_gems.runtime.backend._metax.ops.div").true_divide_
    values = [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0]
    a = torch.tensor(values, device="cuda").repeat(5, 5000)
    b = torch.tensor(
        [0.0, -0.0, float("inf"), float("nan"), 0.73], device="cuda"
    )[:, None]
    expected = a / b
    fn(a, b)
    torch.testing.assert_close(a, expected, atol=1e-6, rtol=2e-6, equal_nan=True)


@pytest.mark.parametrize("kind", ["scalar", "strided", "alias", "fp64"])
def test_row_division_fallback(kind):
    fn = importlib.import_module("flag_gems.runtime.backend._metax.ops.div").true_divide_
    dtype = torch.float64 if kind == "fp64" else torch.float32
    a = torch.randn(3, 32768, device="cuda", dtype=dtype)
    b = torch.rand(3, 1, device="cuda", dtype=dtype) + 0.1
    if kind == "scalar":
        b = 0.73
    elif kind == "strided":
        a = a[:, ::2]
    elif kind == "alias":
        b = a[:, :1]
    expected = a / b
    fn(a, b)
    torch.testing.assert_close(a, expected, atol=1e-6, rtol=2e-6)
