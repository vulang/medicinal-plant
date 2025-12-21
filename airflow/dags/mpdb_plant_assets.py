"""Airflow DAG that reuses the MPDB plant crawler to download list data and photos."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from airflow.decorators import dag, task

# The crawler source lives outside the Airflow DAGs folder. Add a few likely
# locations (repo root and the in-container mount) to PYTHONPATH so imports
# succeed both locally and inside Docker.
_crawler_root_env = os.environ.get("CRAWLER_ROOT")
_CRAWLER_CANDIDATES: Iterable[Path] = (
    Path(_crawler_root_env) if _crawler_root_env else None,
    Path(__file__).resolve().parents[2] / "crawler",
    Path("/opt/crawler"),
)
for candidate in _CRAWLER_CANDIDATES:
    if candidate and candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from plant_crawler import (  # type: ignore  # pylint: disable=wrong-import-position
    INAT_DEFAULT_MAX_PHOTOS,
    INAT_DEFAULT_REQUEST_DELAY,
    _create_session,
    _download_inaturalist_photos,
    _download_mpdb_photos,
    _write_plants_csv,
    crawl_plants,
)


DEFAULT_OUTPUT_ROOT = Path(os.environ.get("PLANT_CRAWLER_OUTPUT_DIR", "/opt/airflow/files"))
DEFAULT_LIMIT = os.environ.get("PLANT_CRAWL_LIMIT")
DEFAULT_DELAY = float(os.environ.get("PLANT_CRAWLER_DELAY", "0.5"))
DEFAULT_INAT_MAX = int(os.environ.get("PLANT_CRAWLER_INAT_MAX_PHOTOS", str(INAT_DEFAULT_MAX_PHOTOS)))
DEFAULT_INAT_DELAY = float(os.environ.get("PLANT_CRAWLER_INAT_DELAY", str(INAT_DEFAULT_REQUEST_DELAY)))


def _output_root() -> Path:
    root = DEFAULT_OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_plants(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    json_path = Path(metadata["json_path"])
    return json.loads(json_path.read_text(encoding="utf-8"))


@dag(
    dag_id="mpdb_plant_assets",
    description="Crawl MPDB plant list and download photos from MPDB + iNaturalist.",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "data-eng",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["mpdb", "plants"],
)
def mpdb_plant_assets():
    @task
    def crawl_plants_task(limit: int | None = None) -> Dict[str, Any]:
        session = _create_session()
        plants, _, _ = crawl_plants(
            limit=limit,
            delay_seconds=max(0.0, DEFAULT_DELAY),
            session=session,
            include_photos=True,
        )
        output_root = _output_root()
        data_dir = output_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "plants.csv"
        _write_plants_csv(csv_path, plants)

        json_path = data_dir / "plants.json"
        json_path.write_text(json.dumps(plants, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "plant_count": len(plants),
        }

    @task
    def download_mpdb_photos(metadata: Dict[str, Any]) -> Dict[str, Any]:
        plants = _load_plants(metadata)
        photo_dir = _output_root() / "photos" / "mpdb"
        _download_mpdb_photos(plants, photo_dir, session=_create_session())
        return {"photo_dir": str(photo_dir), "plants_seen": len(plants)}

    @task
    def download_inat_photos(metadata: Dict[str, Any]) -> Dict[str, Any]:
        plants = _load_plants(metadata)
        photo_dir = _output_root() / "photos" / "inat"
        _download_inaturalist_photos(
            plants,
            photo_dir,
            max_photos=DEFAULT_INAT_MAX,
            request_delay=DEFAULT_INAT_DELAY,
            session=_create_session(),
        )
        return {"photo_dir": str(photo_dir), "plants_seen": len(plants)}

    plant_metadata = crawl_plants_task(limit=int(DEFAULT_LIMIT) if DEFAULT_LIMIT else None)
    plant_metadata >> download_mpdb_photos(plant_metadata)
    plant_metadata >> download_inat_photos(plant_metadata)


dag = mpdb_plant_assets()
