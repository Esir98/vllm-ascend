# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.hisparse_pd import _HiSparsePDReceivingThread, _HiSparsePDWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullConnectorWorker


def test_hisparse_pd_worker_selects_source_and_indexer() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.num_blocks = 8
    worker.kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=32,
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, host_resident=True),
            SimpleNamespace(enable_kv_transfer=True, host_resident=False),
            SimpleNamespace(enable_kv_transfer=False, host_resident=False),
        ],
    )

    assert [worker._is_transfer_group(i) for i in range(3)] == [True, True, False]
    assert [worker._logical_num_blocks(i) for i in range(2)] == [32, 8]


def test_hisparse_pd_worker_slices_replicated_indexer_blocks() -> None:
    worker = _HiSparsePDWorker.__new__(_HiSparsePDWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker.kv_cache_config = SimpleNamespace(
        hisparse_host_num_blocks=4,
        hisparse_host_block_stride=4,
        kv_cache_groups=[
            SimpleNamespace(
                role="hisparse_source", layer_names=["source"], host_resident=True, enable_kv_transfer=True
            ),
            SimpleNamespace(
                role="hisparse_indexer", layer_names=["indexer"], host_resident=False, enable_kv_transfer=True
            ),
        ],
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"indexer": SimpleNamespace(hisparse_block_stride=2)})
    )
    indexer = torch.arange(16).reshape(4, 4)
    source = torch.zeros((4, 4), dtype=torch.uint8)
    with patch.object(MooncakePullConnectorWorker, "register_kv_caches") as register:
        worker.register_kv_caches({"source": source, "indexer": indexer})

    transferred = register.call_args.args[0]["indexer"][0]
    assert torch.equal(transferred, indexer[::2])
    staged_source = register.call_args.args[0]["source"][0]
    assert staged_source.data_ptr() != source.data_ptr()
    staged_source.copy_(torch.arange(16, dtype=torch.uint8).reshape(4, 4))
    thread = _HiSparsePDReceivingThread.__new__(_HiSparsePDReceivingThread)
    thread.host_stage_range = worker._host_stage_range
    thread._after_transfer([staged_source.data_ptr()], [staged_source.nbytes])
    assert torch.equal(source, staged_source)


def test_hisparse_pd_imports_mla_components_by_token() -> None:
    source = torch.zeros((2, 3), dtype=torch.uint8)
    staging = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8)
    thread = _HiSparsePDReceivingThread.__new__(_HiSparsePDReceivingThread)
    thread.host_stage_range = (
        staging.data_ptr(),
        staging.data_ptr() + staging.nbytes,
        source.data_ptr(),
        source.data_ptr() + source.nbytes,
    )
    thread._packed_host_pages = [(staging.data_ptr(), 2, 2, 1)]
    thread._after_transfer([staging.data_ptr(), staging.data_ptr() + 4], [4, 2])
    assert source.tolist() == [[1, 2, 5], [3, 4, 6]]
