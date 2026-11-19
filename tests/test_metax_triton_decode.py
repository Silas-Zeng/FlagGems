# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Paged GQA correctness, padding and replay against an FP32 reference."""

import math

import pytest
import torch

from flag_gems.runtime.backend._metax.ops.triton_attention import try_bf16_decode


def operands(batch, maximum, strided_query=False):
    pages = (maximum + 15) // 16
    cache = torch.randn(batch * pages, 2, 2, 16, 128,
                        device="cuda", dtype=torch.bfloat16)
    k = cache[:, :, 0].transpose(1, 2)
    v = cache[:, :, 1].transpose(1, 2)
    # Non-contiguous physical pages and a table whose row stride exceeds its
    # visible width exercise both page and batch addressing.
    table_storage = torch.empty((batch, pages + 3), device="cuda", dtype=torch.int32)
    table = table_storage[:, :pages]
    table.copy_(torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages))
    q = torch.randn(batch, 1, 16 + (4 if strided_query else 0), 128,
                    device="cuda", dtype=torch.bfloat16)[:, :, :16]
    lengths = torch.linspace(0, maximum, batch, device="cuda").to(torch.int32)
    lengths[-1] = maximum
    out = torch.empty((batch, 1, 16, 128), device="cuda", dtype=q.dtype)
    return q, k, v, out, table, lengths


def reference(q, k, v, table, lengths):
    batch = q.shape[0]
    maximum = table.shape[1] * 16
    keys = k[table.long()].reshape(batch, maximum, 2, 128).float()
    values = v[table.long()].reshape(batch, maximum, 2, 128).float()
    query = q[:, 0].reshape(batch, 2, 8, 128).float()
    scores = torch.einsum("bhgd,blhd->bhgl", query, keys) / math.sqrt(128)
    valid = torch.arange(maximum, device=q.device)[None, :] < lengths[:, None]
    scores.masked_fill_(~valid[:, None, None, :], -float("inf"))
    probability = scores.softmax(-1).nan_to_num()
    return torch.einsum("bhgl,blhd->bhgd", probability, values).reshape_as(q).to(q.dtype)


@pytest.mark.parametrize("batch,maximum,strided", [
    (1, 1, False), (2, 129, True), (3, 17, True), (8, 257, True),
    (33, 257, True), (48, 1025, True), (56, 1025, True),
    (64, 4097, False), (64, 16385, True),
])
def test_paged_decode_fp32(batch, maximum, strided):
    torch.manual_seed(20261118)
    q, k, v, out, table, lengths = operands(batch, maximum, strided)
    assert try_bf16_decode(q, k, v, out, table, lengths, 128 ** -.5)
    torch.testing.assert_close(out, reference(q, k, v, table, lengths), atol=.003, rtol=.02)
    assert torch.isfinite(out).all()
    if batch > 1:
        assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("batch", [1, 33, 64])
def test_decode_graph_reads_updated_inputs_and_lengths(batch):
    torch.manual_seed(20261119)
    q, k, v, out, table, lengths = operands(batch, 1025, True)
    for _ in range(3):
        assert try_bf16_decode(q, k, v, out, table, lengths, 128 ** -.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert try_bf16_decode(q, k, v, out, table, lengths, 128 ** -.5)
    for _ in range(2):
        q.normal_()
        lengths.random_(0, 1026)
        table.copy_(table.roll(1, 1))
        graph.replay()
        torch.testing.assert_close(out, reference(q, k, v, table, lengths), atol=.003, rtol=.02)


def test_unsupported_decode_declines_without_writing():
    q, k, v, out, table, lengths = operands(2, 17)
    out.fill_(37)
    assert not try_bf16_decode(q, k, v, out, table, lengths, 1.0)
    assert torch.all(out == 37)
    assert not try_bf16_decode(q, k.contiguous(), v.contiguous(), out,
                               table, lengths, 128 ** -.5)
    assert torch.all(out == 37)
