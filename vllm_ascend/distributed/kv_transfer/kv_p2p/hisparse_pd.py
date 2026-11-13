# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pull P KV into HiSparse's Host Source and NPU Indexer groups."""

import ctypes
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakeBaseConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakePullConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import as_kv_cache_tensors

_HOST_ALIGNMENT = 2 * 1024 * 1024


class _HiSparsePDReceivingThread(MooncakePullRecvingThread):
    _allow_packed_mla_destination = True

    def _execute_tp_transfer_bucket(self, *args: Any, **kwargs: Any) -> None:
        self._packed_host_pages: list[tuple[int, int, int, int]] = []
        try:
            super()._execute_tp_transfer_bucket(*args, **kwargs)
        finally:
            self._packed_host_pages = []

    def _append_spec_transfer_addresses(self, *args: Any, **kwargs: Any) -> None:
        spec_index = kwargs["spec_index"]
        remote_metadata = kwargs["remote_metadata"]
        remote_tp_metadata = remote_metadata.metadata_by_pcp_rank[kwargs["remote_pcp_rank"]].metadata_by_tp_rank[
            kwargs["remote_tp_rank"]
        ]
        for (local_layer, remote_layer), entries in kwargs["transfer_entries_by_layer"].items():
            local_addrs = self.kv_caches_base_addr[local_layer]
            remote_addrs = remote_tp_metadata.kv_caches_base_addr[remote_layer]
            if len(local_addrs) != 1 or len(remote_addrs) != 2:
                continue
            spec = self.kv_cache_specs[spec_index]
            lengths = remote_metadata.block_lens[remote_layer]
            if len(lengths) != 2 or any(length % spec.block_size for length in lengths):
                raise ValueError("HiSparse PD requires two whole-token MLA components")
            row_lengths = (lengths[0] // spec.block_size, lengths[1] // spec.block_size)
            if sum(lengths) != self.block_lens[local_layer][0]:
                raise ValueError("HiSparse PD Source row size differs from producer MLA")
            for _, local_blocks, _ in entries:
                for block in local_blocks:
                    page = local_addrs[0] + block * self.block_strides[local_layer][0]
                    self._packed_host_pages.append((page, spec.block_size, *row_lengths))
        super()._append_spec_transfer_addresses(*args, **kwargs)

    def _after_transfer(self, local_addresses: list[int], lengths: list[int]) -> None:
        staging_start, staging_end, source_start, source_end = self.host_stage_range
        packed_pages = {page for page, _, _, _ in getattr(self, "_packed_host_pages", [])}
        for address, length in zip(local_addresses, lengths, strict=True):
            if not staging_start <= address < staging_end:
                continue
            destination = source_start + address - staging_start
            if address + length > staging_end or destination + length > source_end:
                raise ValueError("HiSparse PD Host transfer exceeds its registered Source range")
            if any(page <= address < page + length for page in packed_pages):
                continue
            ctypes.memmove(destination, address, length)
        for page, tokens, nope_bytes, rope_bytes in getattr(self, "_packed_host_pages", []):
            destination = source_start + page - staging_start
            page_bytes = tokens * (nope_bytes + rope_bytes)
            if page < staging_start or page + page_bytes > staging_end or destination + page_bytes > source_end:
                raise ValueError("HiSparse PD packed page exceeds its Source range")
            for token in range(tokens):
                row = destination + token * (nope_bytes + rope_bytes)
                ctypes.memmove(row, page + token * nope_bytes, nope_bytes)
                ctypes.memmove(row + nope_bytes, page + tokens * nope_bytes + token * rope_bytes, rope_bytes)


class _HiSparsePDWorker(MooncakePullConnectorWorker):
    _receiving_thread_cls = _HiSparsePDReceivingThread

    def _is_transfer_group(self, group_index: int) -> bool:
        return self.kv_cache_config.kv_cache_groups[group_index].enable_kv_transfer

    def _logical_num_blocks(self, group_index: int) -> int:
        group = self.kv_cache_config.kv_cache_groups[group_index]
        if group.host_resident:
            assert self.kv_cache_config.hisparse_host_num_blocks is not None
            return self.kv_cache_config.hisparse_host_num_blocks
        return self.num_blocks

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        if self.kv_transfer_config.is_kv_consumer:
            context = self.vllm_config.compilation_config.static_forward_context
            caches: dict[str, Any] = dict(kv_caches)
            source_names = [
                name
                for group in self.kv_cache_config.kv_cache_groups
                if group.host_resident and group.enable_kv_transfer
                for name in group.layer_names
            ]
            source_parts = [cache for name in source_names for cache in as_kv_cache_tensors(kv_caches[name])]
            if not source_parts or any(cache.device.type != "cpu" for cache in source_parts):
                raise ValueError("HiSparse PD Source must consist of Host tensors")
            source_start = min(cache.data_ptr() for cache in source_parts)
            source_end = max(
                cache.data_ptr()
                + cache.element_size()
                + sum((size - 1) * stride * cache.element_size() for size, stride in zip(cache.shape, cache.stride()))
                for cache in source_parts
            )
            host_blocks = self.kv_cache_config.hisparse_host_num_blocks
            host_stride = self.kv_cache_config.hisparse_host_block_stride
            assert host_blocks is not None and host_stride is not None
            staging_size = (
                (max(source_end - source_start, host_blocks * host_stride) + _HOST_ALIGNMENT - 1) // _HOST_ALIGNMENT
            ) * _HOST_ALIGNMENT
            owner = torch.empty(staging_size + _HOST_ALIGNMENT - 1, dtype=torch.uint8, pin_memory=True)
            alignment_offset = (-owner.data_ptr()) % _HOST_ALIGNMENT
            staging = owner.narrow(0, alignment_offset, staging_size)
            self._source_staging_owner = owner
            self._host_stage_range = (staging.data_ptr(), staging.data_ptr() + staging_size, source_start, source_end)
            for name in source_names:
                staged_parts = []
                for cache in as_kv_cache_tensors(kv_caches[name]):
                    offset = alignment_offset + cache.data_ptr() - source_start
                    if offset % cache.element_size():
                        raise ValueError(f"Misaligned HiSparse Source layer {name!r}")
                    staged_parts.append(
                        staging.view(cache.dtype).as_strided(
                            cache.shape, cache.stride(), storage_offset=offset // cache.element_size()
                        )
                    )
                caches[name] = tuple(staged_parts)
            for group in self.kv_cache_config.kv_cache_groups:
                if group.role != "hisparse_indexer":
                    continue
                for name in group.layer_names:
                    stride = getattr(context[name], "hisparse_block_stride", 1)
                    if stride > 1:
                        caches[name] = tuple(cache[::stride] for cache in as_kv_cache_tensors(kv_caches[name]))
            kv_caches = caches
        super().register_kv_caches(kv_caches)

    def _configure_receiving_thread(self, thread: MooncakePullRecvingThread) -> None:
        assert isinstance(thread, _HiSparsePDReceivingThread)
        thread.host_stage_range = self._host_stage_range


class HiSparsePDConnector(MooncakeBaseConnector):
    """Use Ascend Mooncake transport with Core HiSparse import ownership."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if self.kv_transfer_config.is_kv_consumer:
            if kv_cache_config.hisparse_host_num_blocks is None:
                raise ValueError("HiSparsePDConnector consumer requires HiSparse Host Source")
            transferable = [group for group in kv_cache_config.kv_cache_groups if group.enable_kv_transfer]
            if (
                len(transferable) != 2
                or not transferable[0].host_resident
                or transferable[1].role != "hisparse_indexer"
            ):
                raise ValueError("HiSparsePDConnector requires Host Source and NPU Indexer transfer groups")
        elif kv_cache_config.hisparse_host_num_blocks is not None:
            raise ValueError("HiSparsePDConnector producer must use ordinary Prefill KV")

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakePullConnectorScheduler(vllm_config, str(self.engine_id), kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = _HiSparsePDWorker(vllm_config, str(self.engine_id), kv_cache_config)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")
