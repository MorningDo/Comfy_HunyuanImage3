"""
Pure validation tests for BlockSwapConfig/BlockSwapStats — no GPU, no model,
no torch.cuda needed (only `import torch` itself, since hunyuan_block_swap.py
imports it at module scope).
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def hunyuan_block_swap():
    return pytest.importorskip("hunyuan_block_swap")


class TestBlockSwapConfig:
    def test_defaults(self, hunyuan_block_swap):
        config = hunyuan_block_swap.BlockSwapConfig()
        assert config.blocks_to_swap == 0
        assert config.prefetch_blocks == 1
        assert config.use_non_blocking is True
        assert config.offload_device == "cpu"
        assert config.swap_start_idx == 0
        assert config.debug is False

    def test_negative_blocks_to_swap_rejected(self, hunyuan_block_swap):
        with pytest.raises(ValueError, match="blocks_to_swap"):
            hunyuan_block_swap.BlockSwapConfig(blocks_to_swap=-1)

    def test_negative_prefetch_blocks_rejected(self, hunyuan_block_swap):
        with pytest.raises(ValueError, match="prefetch_blocks"):
            hunyuan_block_swap.BlockSwapConfig(prefetch_blocks=-1)

    def test_zero_is_valid_for_both(self, hunyuan_block_swap):
        config = hunyuan_block_swap.BlockSwapConfig(blocks_to_swap=0, prefetch_blocks=0)
        assert config.blocks_to_swap == 0
        assert config.prefetch_blocks == 0

    @pytest.mark.parametrize("blocks,prefetch", [(1, 1), (32, 4), (10, 0)])
    def test_valid_combinations(self, hunyuan_block_swap, blocks, prefetch):
        config = hunyuan_block_swap.BlockSwapConfig(blocks_to_swap=blocks, prefetch_blocks=prefetch)
        assert config.blocks_to_swap == blocks
        assert config.prefetch_blocks == prefetch


class TestBlockSwapStats:
    def test_defaults_all_zero(self, hunyuan_block_swap):
        stats = hunyuan_block_swap.BlockSwapStats()
        assert stats.total_swaps_to_gpu == 0
        assert stats.total_swaps_to_cpu == 0
        assert stats.total_swap_time_seconds == 0.0
        assert stats.blocks_currently_on_gpu == 0
        assert stats.blocks_currently_on_cpu == 0

    def test_str_does_not_raise(self, hunyuan_block_swap):
        stats = hunyuan_block_swap.BlockSwapStats(
            total_swaps_to_gpu=3, total_swaps_to_cpu=2, total_swap_time_seconds=1.5,
            blocks_currently_on_gpu=10, blocks_currently_on_cpu=22,
        )
        text = str(stats)
        assert "10" in text and "22" in text
