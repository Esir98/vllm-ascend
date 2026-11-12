# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, sentinel

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


def _cache_view(dtype, device, shape, address, stride):
    cache = MagicMock(dtype=dtype, device=SimpleNamespace(type=device), shape=shape)
    cache.numel.return_value = shape[0] * shape[1] * shape[2]
    cache.element_size.return_value = 2
    cache.stride.return_value = stride
    cache.data_ptr.return_value = address
    cache[0].is_contiguous.return_value = True
    cache[0].numel.return_value = shape[1] * shape[2]
    return cache


@pytest.fixture
def registered_hisparse_worker(monkeypatch):
    dtype = getattr(torch, "bfloat16", sentinel.bfloat16)
    monkeypatch.setattr(torch, "bfloat16", dtype, raising=False)
    source = _cache_view(dtype, "cpu", (3, 128, 2), 0x1040, 384)
    indexer = _cache_view(dtype, "npu", (7, 128, 4), 0x8080, 512)
    indexer.untyped_storage.return_value.data_ptr.return_value = 0x8000
    indexer.untyped_storage.return_value.nbytes.return_value = 0x2000
    host_pool = Mock(nbytes=0x1000)
    host_pool.data_ptr.return_value = 0x1000

    source_name, indexer_name = "model.layers.0.attn", "model.layers.0.indexer"
    resident_name = f"{source_name}.hisparse_resident"
    hot_name = f"{source_name}.hisparse_hot"
    groups = [
        SimpleNamespace(role=SimpleNamespace(value="hisparse_source"), layer_names=[source_name]),
        SimpleNamespace(role=SimpleNamespace(value="hisparse_indexer"), layer_names=[indexer_name]),
        SimpleNamespace(enable_kv_transfer=False, layer_names=[resident_name]),
        SimpleNamespace(enable_kv_transfer=False, layer_names=[hot_name]),
    ]
    context = {
        source_name: SimpleNamespace(
            hisparse_cache=SimpleNamespace(runtime=SimpleNamespace(registered_host_pool=host_pool))
        ),
        indexer_name: SimpleNamespace(hisparse_block_stride=2),
    }
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4, hisparse_host_num_blocks=3, hisparse_shared_host_pool=False, kv_cache_groups=groups
    )
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context))
    worker.hf_config = SimpleNamespace(num_hidden_layers=1)
    worker.num_layers = 1
    worker.use_hisparse_store = True
    worker.use_layerwise = False
    worker.load_async = True
    worker.cacheable_group_ids = [0, 1]
    worker.grouped_block_size = [128] * len(groups)
    worker.token_database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, group_id) for group_id in range(len(groups))],
        worker.grouped_block_size,
        None,
    )
    worker.m_store = Mock(spec=MemcacheBackend)
    worker._start_kv_transfer_threads = Mock()

    # Core keeps the Resident view and drops the Hot view before registration.
    worker.register_kv_caches({source_name: (source,), indexer_name: (indexer,), resident_name: sentinel.resident})
    return worker, host_pool


def test_hisparse_bulk_registers_source_and_indexer_with_separate_page_geometry(registered_hisparse_worker):
    worker, host_pool = registered_hisparse_worker

    worker.m_store.register_hisparse_buffers.assert_called_once_with(host_pool, {0x8000: 0x2000})
    worker._start_kv_transfer_threads.assert_called_once_with()
    assert worker.group_kv_caches_base_addr == {0: [0x1040], 1: [0x8080]}
    assert worker.group_block_len == {0: [512], 1: [1024]}
    assert worker.group_block_stride == {0: [768], 1: [2048]}
    assert worker.group_kv_cache_families == {0: "hisparse_source", 1: "hisparse_indexer"}


@pytest.mark.parametrize(("get_results", "failed"), [([0, 0], False), ([0, -1], True), ([0], True)])
def test_hisparse_bulk_receive_requires_complete_prefix(registered_hisparse_worker, get_results, failed):
    worker, _ = registered_hisparse_worker
    worker.m_store.get.return_value = get_results
    receiver = KVCacheStoreRecvingThread(
        m_store=worker.m_store,
        token_database=worker.token_database,
        block_size=worker.grouped_block_size,
        tp_rank=0,
        require_complete_load=True,
    )
    request = ReqMeta(
        req_id="load",
        block_ids_by_group=[[2], [3], [], []],
        block_hashes=[b"prefix"],  # type: ignore[list-item]
        kv_cache_group_ids=[0, 1],
        load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=128, can_load=True, token_len=128),
        can_save=True,
    )

    receiver.request_queue.put(request)
    receiver._handle_request(request)

    worker.m_store.get.assert_called_once()
    keys, addresses, sizes = worker.m_store.get.call_args.args
    assert len(keys) == 2
    assert addresses == [[0x1640], [0x9880]]
    assert sizes == [[512], [1024]]
    assert receiver.get_and_clear_finished_requests() == (set() if failed else {"load"})
    assert receiver.get_and_clear_failed_requests() == ({"load"} if failed else set())
    assert request.can_save is (not failed)
    assert receiver.request_queue.unfinished_tasks == 0


def test_memcache_bulk_splits_host_and_device_descriptors_and_preserves_result_order():
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend._hisparse_buffers = None
    backend._lazy_init = False
    backend._store_initialized = True
    backend._pending_buffers = None
    backend.store = Mock()
    backend.store.register_buffer.return_value = 0
    host = Mock(device=SimpleNamespace(type="cpu"), nbytes=0x1000)
    host.data_ptr.return_value = 0x1000
    host.is_contiguous.return_value = True
    host.is_pinned.return_value = True

    backend.register_hisparse_buffers(host, {0x8000: 0x2000})

    assert backend.store.register_buffer.call_args_list == [call(0x1000, 0x1000), call(0x8000, 0x2000)]
    keys = ["source", "indexer"]
    addresses = [[0x1640], [0x9880]]
    sizes = [[512], [1024]]
    backend.store.batch_get_into_layers.side_effect = [[0], [-7]]
    backend.store.batch_put_from_layers.side_effect = [[-8], [0]]

    assert backend.get(keys, addresses, sizes) == [-7, 0]
    assert backend.put(keys, addresses, sizes) == [True, False]
    assert backend.store.batch_get_into_layers.call_args_list == [
        call(["indexer"], [addresses[1]], [sizes[1]], MmcDirect.COPY_G2L.value),
        call(["source"], [addresses[0]], [sizes[0]], MmcDirect.COPY_G2H.value),
    ]
    assert backend.store.batch_put_from_layers.call_args_list == [
        call(["indexer"], [addresses[1]], [sizes[1]], MmcDirect.COPY_L2G.value),
        call(["source"], [addresses[0]], [sizes[0]], MmcDirect.COPY_H2G.value),
    ]

    backend.store.batch_get_into_layers.reset_mock()
    assert backend.get(["mixed"], [[0x1640, 0x9880]], [[512, 1024]]) is None
    backend.store.batch_get_into_layers.assert_not_called()
