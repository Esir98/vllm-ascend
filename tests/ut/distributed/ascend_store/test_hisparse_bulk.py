# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta, infer_cacheable_group_ids


def test_bulk_lookup_preserves_group_ids_and_excludes_local_cache_groups():
    cacheable = SimpleNamespace(prefix_cacheable=True)
    groups = [
        SimpleNamespace(kv_cache_spec=cacheable, enable_kv_transfer=True),
        SimpleNamespace(kv_cache_spec=cacheable, enable_kv_transfer=False),
        SimpleNamespace(kv_cache_spec=cacheable, enable_kv_transfer=False),
        SimpleNamespace(kv_cache_spec=cacheable, enable_kv_transfer=True),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_cacheable=False), enable_kv_transfer=True),
    ]

    assert infer_cacheable_group_ids(groups) == [0, 3]


def test_bulk_save_retains_host_and_device_blocks_until_all_workers_finish():
    device_pool = Mock(blocks=[object() for _ in range(10)])
    host_pool = Mock(blocks=[object() for _ in range(10)])
    tracker = InFlightSaveBlockTracker(expected_worker_count=3)
    req_meta = ReqMeta(
        req_id="save",
        block_ids_by_group=[[2, 1, 9], [7, 8], [1, 0, 2, 3, 8]],
        can_save=True,
        kv_cache_group_ids=[0, 2],
        save_start_token=128,
        save_end_token=256,
        target_token_len=300,
    )

    tracker.retain_hisparse(req_meta, 0, [128, 128, 64], device_pool, host_pool)

    # Retain full source pages for PUT retries, including pages before the
    # current save start. The private group, null page and partial tails
    # are not read by the save.
    host_pages = [host_pool.blocks[2], host_pool.blocks[1]]
    device_pages = [device_pool.blocks[1], device_pool.blocks[2], device_pool.blocks[3]]
    host_pool.touch.assert_called_once_with(host_pages)
    device_pool.touch.assert_called_once_with(device_pages)
    event_id = req_meta.event_id
    assert event_id is not None
    assert tracker.has_pending_work()

    tracker.complete({event_id: 1}, device_pool, host_pool)

    assert tracker.has_pending_work()
    host_pool.free_blocks.assert_not_called()
    device_pool.free_blocks.assert_not_called()

    tracker.complete({event_id: 2}, device_pool, host_pool)

    host_pool.free_blocks.assert_called_once_with(host_pages[::-1])
    device_pool.free_blocks.assert_called_once_with(device_pages[::-1])
    assert not tracker.has_pending_work()
    assert not tracker.has_tracked_blocks()


def test_bulk_load_only_metadata_does_not_retain_save_blocks():
    tracker = InFlightSaveBlockTracker(expected_worker_count=1)
    req_meta = ReqMeta(req_id="load", can_save=False)
    device_pool = Mock()
    host_pool = Mock()

    tracker.retain_hisparse(req_meta, 0, [128], device_pool, host_pool)

    assert req_meta.event_id is None
    assert not tracker.has_pending_work()
    device_pool.touch.assert_not_called()
    host_pool.touch.assert_not_called()
