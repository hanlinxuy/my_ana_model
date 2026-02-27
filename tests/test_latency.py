"""Tests for LatencyCalculator."""

import pytest
from moe_simulator.core.latency import BandwidthModel, LatencyCalculator


class TestBandwidthModelBasics:
    """Test BandwidthModel basic operations."""

    def test_default_values(self):
        """Test default bandwidth values."""
        bw = BandwidthModel()
        assert bw.flash_to_ddr == 1.0
        assert bw.ddr_to_npu == 8.0
        assert bw.flash_to_pim == 4.0

    def test_custom_values(self):
        """Test custom bandwidth values."""
        bw = BandwidthModel(
            flash_to_ddr=2.0,
            ddr_to_npu=16.0,
            flash_to_pim=8.0,
        )
        assert bw.flash_to_ddr == 2.0
        assert bw.ddr_to_npu == 16.0
        assert bw.flash_to_pim == 8.0

    def test_invalid_negative_flash_to_ddr(self):
        """Test negative flash_to_ddr raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            BandwidthModel(flash_to_ddr=-1.0)

    def test_invalid_negative_ddr_to_npu(self):
        """Test negative ddr_to_npu raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            BandwidthModel(ddr_to_npu=-1.0)

    def test_invalid_negative_flash_to_pim(self):
        """Test negative flash_to_pim raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            BandwidthModel(flash_to_pim=-1.0)


class TestBandwidthModelSerialization:
    """Test BandwidthModel serialization."""

    def test_to_dict(self):
        """Test to_dict method."""
        bw = BandwidthModel(flash_to_ddr=2.0, ddr_to_npu=16.0, flash_to_pim=8.0)
        result = bw.to_dict()
        assert result == {
            "flash_to_ddr": 2.0,
            "ddr_to_npu": 16.0,
            "flash_to_pim": 8.0,
        }

    def test_from_dict(self):
        """Test from_dict method."""
        data = {"flash_to_ddr": 2.0, "ddr_to_npu": 16.0, "flash_to_pim": 8.0}
        bw = BandwidthModel.from_dict(data)
        assert bw.flash_to_ddr == 2.0
        assert bw.ddr_to_npu == 16.0
        assert bw.flash_to_pim == 8.0

    def test_from_dict_defaults(self):
        """Test from_dict with missing keys uses defaults."""
        bw = BandwidthModel.from_dict({})
        assert bw.flash_to_ddr == 1.0
        assert bw.ddr_to_npu == 8.0
        assert bw.flash_to_pim == 4.0


class TestLatencyCalculatorBasics:
    """Test LatencyCalculator basic operations."""

    def test_default_values(self):
        """Test default latency calculator values."""
        calc = LatencyCalculator()
        assert calc.k1 == 1.0
        assert calc.k2 == 1.0
        assert calc.k3 == 1.0

    def test_custom_coefficients(self):
        """Test custom coefficients."""
        calc = LatencyCalculator(k1=2.0, k2=3.0, k3=1.5)
        assert calc.k1 == 2.0
        assert calc.k2 == 3.0
        assert calc.k3 == 1.5


class TestLatencyCalculation:
    """Test latency calculation formula."""

    def test_calculate_basic(self, latency_calculator):
        """Test basic latency calculation."""
        result = latency_calculator.calculate(k1=1.0, k2=1.0, k3=1.0)
        assert result > 0

    def test_calculate_k1_dominant(self, latency_calculator):
        """Test when k1 is dominant (cache miss)."""
        result = latency_calculator.calculate(k1=10.0, k2=1.0, k3=1.0)
        expected = 10.0 * (1.0 + 8.0)
        assert result == expected

    def test_calculate_k2_dominant(self, latency_calculator):
        """Test when k2 is dominant (PIM path)."""
        result = latency_calculator.calculate(k1=1.0, k2=10.0, k3=1.0)
        expected = 10.0 * 4.0
        assert result == expected

    def test_calculate_k3_dominant(self, latency_calculator):
        """Test when k3 is dominant (cache hit)."""
        result = latency_calculator.calculate(k1=1.0, k2=1.0, k3=10.0)
        expected = 10.0 * 8.0
        assert result == expected


class TestLatencyBoundaryCases:
    """Test boundary cases for latency calculation."""

    def test_all_zero(self, latency_calculator):
        """Test all zero coefficients."""
        result = latency_calculator.calculate(k1=0.0, k2=0.0, k3=0.0)
        assert result == 0.0

    def test_k1_zero(self, latency_calculator):
        """Test k1 = 0."""
        result = latency_calculator.calculate(k1=0.0, k2=1.0, k3=1.0)
        assert result == max(0.0, 1.0 * 4.0, 1.0 * 8.0)

    def test_k2_zero(self, latency_calculator):
        """Test k2 = 0."""
        result = latency_calculator.calculate(k1=1.0, k2=0.0, k3=1.0)
        assert result == max(1.0 * 9.0, 0.0, 1.0 * 8.0)

    def test_k3_zero(self, latency_calculator):
        """Test k3 = 0."""
        result = latency_calculator.calculate(k1=1.0, k2=1.0, k3=0.0)
        assert result == max(1.0 * 9.0, 1.0 * 4.0, 0.0)

    def test_calculate_boundary_k1_zero(self, latency_calculator):
        """Test calculate_boundary with k1=0."""
        result = latency_calculator.calculate_boundary(k1=0.0, k2=1.0, k3=1.0)
        assert result == 8.0

    def test_calculate_boundary_all_zero(self, latency_calculator):
        """Test calculate_boundary with all zeros returns 0."""
        result = latency_calculator.calculate_boundary(k1=0.0, k2=0.0, k3=0.0)
        assert result == 0.0

    def test_calculate_boundary_epsilon(self, latency_calculator):
        """Test calculate_boundary with very small values."""
        result = latency_calculator.calculate_boundary(k1=1e-10, k2=1e-10, k3=1e-10)
        assert result == 0.0


class TestLatencyComponents:
    """Test get_latency_components method."""

    def test_get_components(self, latency_calculator):
        """Test getting individual latency components."""
        components = latency_calculator.get_latency_components(k1=1.0, k2=1.0, k3=1.0)
        assert "cache_miss" in components
        assert "pim" in components
        assert "cached" in components

    def test_components_values(self, latency_calculator):
        """Test component values match expected formula."""
        components = latency_calculator.get_latency_components(k1=2.0, k2=1.5, k3=3.0)
        assert components["cache_miss"] == 2.0 * 9.0
        assert components["cached"] == 3.0 * 8.0


class TestLatencySetCoefficients:
    """Test set_coefficients method."""

    def test_set_all(self, latency_calculator):
        """Test setting all coefficients."""
        latency_calculator.set_coefficients(k1=5.0, k2=6.0, k3=7.0)
        assert latency_calculator.k1 == 5.0
        assert latency_calculator.k2 == 6.0
        assert latency_calculator.k3 == 7.0

    def test_set_partial(self, latency_calculator):
        """Test setting only some coefficients."""
        latency_calculator.set_coefficients(k1=5.0)
        assert latency_calculator.k1 == 5.0
        assert latency_calculator.k2 == 1.0
        assert latency_calculator.k3 == 1.0


class TestLatencyCalculateWithDefaults:
    """Test calculate_with_defaults method."""

    def test_calculate_with_defaults(self, latency_calculator):
        """Test using stored default coefficients."""
        latency_calculator.k1 = 2.0
        latency_calculator.k2 = 3.0
        latency_calculator.k3 = 1.0
        result = latency_calculator.calculate_with_defaults()
        expected = latency_calculator.calculate(2.0, 3.0, 1.0)
        assert result == expected


class TestLatencyCustomBandwidth:
    """Test latency calculation with custom bandwidth."""

    def test_custom_bandwidth_k1(self, latency_calculator_custom):
        """Test with custom bandwidth - k1 dominant."""
        result = latency_calculator_custom.calculate(k1=10.0, k2=1.0, k3=1.0)
        expected = 10.0 * (2.0 + 16.0)
        assert result == expected

    def test_custom_bandwidth_k2(self, latency_calculator_custom):
        """Test with custom bandwidth - k2 dominant."""
        result = latency_calculator_custom.calculate(k1=1.0, k2=10.0, k3=1.0)
        expected = 10.0 * 8.0
        assert result == expected


class TestLatencyRepr:
    """Test string representation."""

    def test_repr(self, latency_calculator):
        """Test __repr__ method."""
        repr_str = repr(latency_calculator)
        assert "LatencyCalculator" in repr_str
        assert "k1=" in repr_str
        assert "k2=" in repr_str
        assert "k3=" in repr_str
