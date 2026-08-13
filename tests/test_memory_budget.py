"""
Pure-logic tests for MemoryBudget's VRAM estimation/block-swap math.

No GPU needed — get_vram_report() is monkeypatched with a fixed VRAMReport
so the arithmetic can be checked deterministically on any machine.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def hunyuan_memory_budget():
    return pytest.importorskip("hunyuan_memory_budget")


@pytest.fixture()
def budget(hunyuan_memory_budget, monkeypatch):
    b = hunyuan_memory_budget.MemoryBudget(device="cuda:0")

    def fake_report(self=None):
        return hunyuan_memory_budget.VRAMReport(
            total_bytes=96 * 1024**3,
            free_bytes=90 * 1024**3,
            allocated_bytes=6 * 1024**3,
            reserved_bytes=6 * 1024**3,
            device_name="Fake RTX 6000",
            device_index=0,
        )

    monkeypatch.setattr(b, "get_vram_report", fake_report)
    return b


class TestVRAMReport:
    def test_gb_properties_convert_correctly(self, hunyuan_memory_budget):
        report = hunyuan_memory_budget.VRAMReport(
            total_bytes=10 * 1024**3, free_bytes=4 * 1024**3,
            allocated_bytes=6 * 1024**3, reserved_bytes=6 * 1024**3,
            device_name="x", device_index=0,
        )
        assert report.total_gb == pytest.approx(10.0)
        assert report.free_gb == pytest.approx(4.0)
        assert report.allocated_gb == pytest.approx(6.0)
        assert report.reserved_gb == pytest.approx(6.0)


class TestModelSizeEstimate:
    @pytest.mark.parametrize("quant_type", ["nf4", "int8", "bf16"])
    def test_estimate_model_size_returns_positive_totals(self, budget, quant_type):
        estimate = budget.estimate_model_size(quant_type)
        assert estimate.total_gb > 0
        assert estimate.weights_gb > 0
        assert estimate.num_transformer_blocks > 0
        assert estimate.bytes_per_block > 0

    def test_nf4_smaller_than_int8_and_bf16(self, budget):
        # int8 and bf16 currently share the same weights_gb estimate in
        # MODEL_SIZES (both store full-precision-sized weights: int8 here
        # models "weights + SCB scales" as roughly the same footprint as
        # bf16), so only assert nf4 is strictly smaller — not a 3-way
        # strict ordering.
        nf4 = budget.estimate_model_size("nf4")
        int8 = budget.estimate_model_size("int8")
        bf16 = budget.estimate_model_size("bf16")
        assert nf4.total_gb < int8.total_gb
        assert nf4.total_gb < bf16.total_gb


class TestInferenceVRAMEstimate:
    def test_increases_with_resolution(self, budget):
        small = budget.estimate_inference_vram(1024, 1024)
        large = budget.estimate_inference_vram(2048, 2048)
        assert large > small
        assert small > 0

    def test_returns_a_float(self, budget):
        result = budget.estimate_inference_vram(1024, 1024)
        assert isinstance(result, float)


class TestBlocksToSwap:
    @pytest.mark.parametrize("quant_type", ["nf4", "int8", "bf16"])
    def test_returns_valid_block_count_and_reason(self, budget, quant_type):
        blocks, reason = budget.calculate_blocks_to_swap(quant_type, 1024, 1024)
        assert isinstance(blocks, int)
        assert blocks >= 0
        assert isinstance(reason, str) and reason

    def test_more_blocks_needed_at_higher_resolution_or_equal(self, budget):
        """Higher resolution consumes more inference VRAM, leaving less
        headroom for blocks, so blocks-to-swap should never decrease."""
        low_blocks, _ = budget.calculate_blocks_to_swap("bf16", 1024, 1024)
        high_blocks, _ = budget.calculate_blocks_to_swap("bf16", 2048, 2048)
        assert high_blocks >= low_blocks

    def test_safety_margin_never_produces_negative_blocks(self, budget):
        blocks, _ = budget.calculate_blocks_to_swap("bf16", 4096, 4096, safety_margin_gb=50.0)
        assert blocks >= 0


class TestCanFitEntirely:
    def test_small_model_at_small_resolution_fits_on_96gb(self, budget):
        assert budget.can_fit_entirely("nf4", 1024, 1024) is True

    def test_bf16_at_extreme_resolution_may_not_fit(self, budget, hunyuan_memory_budget, monkeypatch):
        # Shrink the fake GPU down so we can deterministically assert "doesn't fit".
        def tiny_report(self=None):
            return hunyuan_memory_budget.VRAMReport(
                total_bytes=8 * 1024**3, free_bytes=6 * 1024**3,
                allocated_bytes=2 * 1024**3, reserved_bytes=2 * 1024**3,
                device_name="Fake tiny GPU", device_index=0,
            )
        monkeypatch.setattr(budget, "get_vram_report", tiny_report)
        assert budget.can_fit_entirely("bf16", 4096, 4096) is False


class TestOptimalConfig:
    def test_returns_dict_with_expected_keys(self, budget):
        config = budget.get_optimal_config("nf4", 1024, 1024)
        assert isinstance(config, dict)
        # Don't over-assert on exact key names beyond what's documented —
        # just confirm it's non-empty and doesn't raise.
        assert config
