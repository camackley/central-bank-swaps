from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cbs.config.banks import BankConfig, load_bank_config

FIXTURES = Path(__file__).parent.parent / "fixtures" / "config"


class TestLoadValidBankConfig:
    """Test loading a valid YAML config with all 10 banks."""

    def test_parses_all_10_banks(self) -> None:
        config = load_bank_config(FIXTURES / "valid_banks.yaml")
        assert len(config.banks) == 10

    def test_bank_has_required_fields(self) -> None:
        config = load_bank_config(FIXTURES / "valid_banks.yaml")
        fed = config.banks[0]
        assert fed.name == "Federal Reserve"
        assert fed.country == "US"
        assert str(fed.homepage_url) == "https://www.federalreserve.gov/"

    def test_bank_names_are_unique(self) -> None:
        config = load_bank_config(FIXTURES / "valid_banks.yaml")
        names = [b.name for b in config.banks]
        assert len(names) == len(set(names))


class TestBankConfigRequiresNameCountryUrl:
    """Test that name, country, and homepage_url are required."""

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            load_bank_config(FIXTURES / "missing_name.yaml")

    def test_missing_country_raises(self) -> None:
        with pytest.raises(ValidationError):
            load_bank_config(FIXTURES / "missing_country.yaml")

    def test_missing_homepage_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            load_bank_config(FIXTURES / "missing_url.yaml")


class TestBankConfigOptionalPressReleasesUrl:
    """Test that press_releases_url is optional."""

    def test_bank_without_press_releases_url(self) -> None:
        config = load_bank_config(FIXTURES / "valid_banks.yaml")
        # PBoC has no press_releases_url
        pboc = next(b for b in config.banks if b.name == "People's Bank of China")
        assert pboc.press_releases_url is None

    def test_bank_with_press_releases_url(self) -> None:
        config = load_bank_config(FIXTURES / "valid_banks.yaml")
        fed = config.banks[0]
        assert fed.press_releases_url is not None
        assert "pressreleases" in str(fed.press_releases_url)


class TestBankConfigDefaultTimeout:
    """Test that page_load_timeout defaults to 30s."""

    def test_default_timeout_is_30(self) -> None:
        bank = BankConfig(
            name="Test Bank",
            country="Testland",
            homepage_url="https://example.com",
        )
        assert bank.page_load_timeout == 30

    def test_explicit_timeout_overrides_default(self) -> None:
        bank = BankConfig(
            name="Test Bank",
            country="Testland",
            homepage_url="https://example.com",
            page_load_timeout=60,
        )
        assert bank.page_load_timeout == 60


class TestHistoricalCutoffYear:
    """Test that historical_cutoff_year defaults to 2008 and can be overridden."""

    def test_default_cutoff_year_is_2008(self) -> None:
        bank = BankConfig(
            name="Test Bank",
            country="Testland",
            homepage_url="https://example.com",
        )
        assert bank.historical_cutoff_year == 2008

    def test_explicit_cutoff_year_overrides_default(self) -> None:
        bank = BankConfig(
            name="Test Bank",
            country="Testland",
            homepage_url="https://example.com",
            historical_cutoff_year=2000,
        )
        assert bank.historical_cutoff_year == 2000


class TestProductionConfig:
    """Test that the production config/banks.yaml loads correctly."""

    def test_production_config_loads(self) -> None:
        prod_path = Path(__file__).parent.parent.parent / "config" / "banks.yaml"
        config = load_bank_config(prod_path)
        assert len(config.banks) == 10


class TestInvalidYamlRaisesError:
    """Test that malformed YAML raises a clear error."""

    def test_invalid_yaml_syntax(self) -> None:
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            load_bank_config(FIXTURES / "invalid.yaml")

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_bank_config(FIXTURES / "nonexistent.yaml")
