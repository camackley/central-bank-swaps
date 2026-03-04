"""Central bank configuration loader — Pydantic models + YAML parsing."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, HttpUrl


class BankConfig(BaseModel):
    """Configuration for a single central bank."""

    name: str
    country: str
    homepage_url: HttpUrl
    press_releases_url: HttpUrl | None = None
    page_load_timeout: int = 30
    historical_cutoff_year: int = 2008


class BanksConfig(BaseModel):
    """Top-level configuration containing all central banks."""

    banks: list[BankConfig]


def load_bank_config(path: Path) -> BanksConfig:
    """Load and validate central bank configuration from a YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is syntactically invalid.
        pydantic.ValidationError: If the data fails schema validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            msg = f"Invalid YAML in {path}: {exc}"
            raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"Invalid config structure in {path}: expected a mapping"
        raise ValueError(msg)

    return BanksConfig.model_validate(raw)
