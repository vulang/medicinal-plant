from __future__ import annotations

from plant_crawler import (
    BASE_URL,
    PLANT_LIST_URL,
    crawl_cultivation,
    crawl_plants as _crawl_plants,
    crawl_tissue_culture_literature,
    fetch_cultivation_data,
    fetch_plant_detail,
    fetch_tissue_culture_literature,
    iter_plant_entry_urls,
)


def crawl_plants(*args, **kwargs):
    """Backwards-compatible wrapper returning only plant records."""
    plant_entries, _, _ = _crawl_plants(*args, **kwargs)
    return plant_entries


def crawl_plants_with_extras(*args, **kwargs):
    """Return plant, tissue literature, and cultivation datasets."""
    return _crawl_plants(*args, **kwargs)

__all__ = [
    "BASE_URL",
    "PLANT_LIST_URL",
    "crawl_plants",
    "crawl_plants_with_extras",
    "crawl_tissue_culture_literature",
    "crawl_cultivation",
    "fetch_plant_detail",
    "fetch_cultivation_data",
    "fetch_tissue_culture_literature",
    "iter_plant_entry_urls",
]
