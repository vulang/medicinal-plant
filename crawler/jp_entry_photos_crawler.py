#!/usr/bin/env python3
"""
Dedicated crawler for MPDB JP identification entries with photo download support.

The JP identification category does not expose a list endpoint, so this crawler
walks the known identifier range and fetches each detail page directly. It also
locates associated photo links so the assets can be downloaded alongside the
structured data.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "http://mpdb.nibiohn.go.jp"
DETAIL_ENDPOINT = "/mpdb-bin/view_jp_identification_data.cgi"
PHOTO_URL_PATTERN = re.compile(
    r"get_image\.cgi\?table=jp_identification_photo_data&column=jp_identification_photo_file&id=(\d+)",
    re.IGNORECASE,
)
DEFAULT_START_ID = 1
DEFAULT_STOP_ID = 676
DEFAULT_DELAY = 0.5
DEFAULT_TIMEOUT = 30.0
DEFAULT_LANG = "en"

logger = logging.getLogger(__name__)


def normalise_text(value: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    return " ".join(value.split())


@dataclass
class JPIdentificationPhoto:
    photo_id: str
    full_size_url: str
    thumbnail_url: Optional[str] = None
    file_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "photo_id": self.photo_id,
            "full_size_url": self.full_size_url,
        }
        if self.thumbnail_url:
            payload["thumbnail_url"] = self.thumbnail_url
        if self.file_name:
            payload["file_name"] = self.file_name
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload


@dataclass
class JPIdentificationEntry:
    jp_id: int
    url: str
    label: Optional[str]
    fields: Dict[str, str]
    photos: List[JPIdentificationPhoto] = field(default_factory=list)
    memos: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "jp_id": self.jp_id,
            "url": self.url,
            "label": self.label,
            "fields": dict(self.fields),
        }
        if self.photos:
            payload["photos"] = [photo.to_dict() for photo in self.photos]
        if self.memos:
            payload["memos"] = list(self.memos)
        return payload


class JPIdentificationCrawler:
    """Crawler specialised for JP identification entries."""

    def __init__(
        self,
        *,
        lang: str = DEFAULT_LANG,
        delay: float = DEFAULT_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        download_photos: bool = False,
        photo_root: Optional[Path] = None,
    ):
        self.lang = lang
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.download_photos = download_photos
        self.photo_root = Path(photo_root) if photo_root else Path("crawler/photos/jp_identification")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "jp-identification-crawler/1.0 (+https://github.com/)",
                "Accept-Language": lang,
                "Accept-Charset": "utf-8",
            }
        )

    def crawl(
        self,
        *,
        start: int = DEFAULT_START_ID,
        stop: int = DEFAULT_STOP_ID,
        limit: Optional[int] = None,
        id_iterable: Optional[Iterable[int]] = None,
    ) -> List[JPIdentificationEntry]:
        """
        Crawl the JP identification detail pages within the provided range.

        Args:
            start: First identifier to request (inclusive).
            stop: Last identifier to request (inclusive).
            limit: Optional cap on the number of entries to collect.
            id_iterable: Optional explicit iterable of IDs to crawl, overriding start/stop.
        """
        ids: Iterable[int]
        if id_iterable is not None:
            ids = id_iterable
        else:
            if stop < start:
                raise ValueError("stop must be greater than or equal to start")
            ids = range(start, stop + 1)

        entries: List[JPIdentificationEntry] = []
        for identifier in ids:
            if limit is not None and len(entries) >= limit:
                break
            url = f"{BASE_URL}{DETAIL_ENDPOINT}?id={identifier}&lang={self.lang}"
            try:
                html = self._fetch(url)
            except requests.RequestException as exc:
                logger.warning("Skipping JP identification %s due to HTTP error: %s", identifier, exc)
                self._sleep()
                continue
            entry = self._parse_entry(identifier, url, html)
            if entry is None:
                logger.debug("No content found for JP identification %s", identifier)
                self._sleep()
                continue
            if self.download_photos and entry.photos:
                self._download_photos(entry)
            entries.append(entry)
            self._sleep()
        return entries

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def _fetch(self, url: str) -> str:
        """Fetch raw HTML and decode it using a sensible default encoding."""
        logger.debug("Fetching %s", url)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        encoding = response.encoding or response.apparent_encoding or "utf-8"
        if encoding.lower() == "iso-8859-1":
            encoding = "utf-8"
        try:
            return response.content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            logger.debug("Falling back to utf-8 decoding for %s", url)
            return response.content.decode("utf-8", errors="replace")

    def _parse_entry(self, identifier: int, url: str, html: str) -> Optional[JPIdentificationEntry]:
        soup = BeautifulSoup(html, "html.parser")

        tables = list(soup.find_all("table", class_="detail"))
        if not tables:
            tables = [tbl for tbl in soup.find_all("table") if tbl.find("tr")]
        if not tables:
            fallback_text = normalise_text(soup.get_text(" ", strip=True))
            if not fallback_text:
                return None
            return JPIdentificationEntry(
                jp_id=identifier,
                url=url,
                label=None,
                fields={"Content": fallback_text},
            )

        fields: Dict[str, str] = {}
        photos: List[JPIdentificationPhoto] = []
        memos: List[str] = []
        row_counter = 1
        seen_photo_ids: set[str] = set()

        for table in tables:
            table_last_photo: Optional[JPIdentificationPhoto] = None
            for tr in table.find_all("tr", recursive=False):
                cells = tr.find_all("td", recursive=False)
                if not cells:
                    continue

                if len(cells) == 1:
                    text = normalise_text(cells[0].get_text(" ", strip=True))
                    if text:
                        key = f"Field {row_counter}"
                        fields[key] = text
                        row_counter += 1
                    continue

                label = normalise_text(cells[0].get_text(" ", strip=True))
                value_cell = cells[1]
                label_normalized = re.sub(r"\s+", " ", label.lower())

                if "jp identification photo file" in label_normalized:
                    extracted = self._extract_photos(value_cell)
                    if extracted:
                        for photo in extracted:
                            if photo.photo_id in seen_photo_ids:
                                continue
                            photos.append(photo)
                            seen_photo_ids.add(photo.photo_id)
                        table_last_photo = photos[-1]
                    continue

                if "jp identification memo" in label_normalized:
                    memo_text = normalise_text(value_cell.get_text(" ", strip=True))
                    if memo_text:
                        target = table_last_photo or (photos[-1] if photos else None)
                        if target is not None:
                            target.notes.append(memo_text)
                        else:
                            memos.append(memo_text)
                    continue

                value_text = self._cell_to_text(value_cell)
                if label:
                    if label in fields and value_text:
                        fields[label] = f"{fields[label]} | {value_text}"
                    elif value_text:
                        fields[label] = value_text
                elif value_text:
                    fields[f"Field {row_counter}"] = value_text
                    row_counter += 1

        entry_label = self._derive_label(fields)
        return JPIdentificationEntry(
            jp_id=identifier,
            url=url,
            label=entry_label,
            fields=fields,
            photos=photos,
            memos=memos,
        )

    def _extract_photos(self, cell) -> List[JPIdentificationPhoto]:
        photos: List[JPIdentificationPhoto] = []
        seen_ids: set[str] = set()

        for anchor in cell.find_all("a", href=True):
            href = anchor.get("href") or ""
            match = PHOTO_URL_PATTERN.search(href)
            if not match:
                continue
            photo_id = match.group(1)
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)
            full_url = urljoin(BASE_URL, href)
            img_tag = anchor.find("img")
            thumb_url = urljoin(BASE_URL, img_tag["src"]) if img_tag and img_tag.get("src") else None
            file_name = f"{photo_id}.jpg"
            photos.append(
                JPIdentificationPhoto(
                    photo_id=photo_id,
                    full_size_url=full_url,
                    thumbnail_url=thumb_url,
                    file_name=file_name,
                )
            )

        if not photos:
            for img in cell.find_all("img", src=True):
                src = img.get("src") or ""
                match = PHOTO_URL_PATTERN.search(src)
                if not match:
                    continue
                photo_id = match.group(1)
                if photo_id in seen_ids:
                    continue
                seen_ids.add(photo_id)
                photo_url = urljoin(BASE_URL, src)
                photos.append(
                    JPIdentificationPhoto(
                        photo_id=photo_id,
                        full_size_url=photo_url,
                        thumbnail_url=photo_url,
                        file_name=f"{photo_id}.jpg",
                    )
                )

        return photos

    def _cell_to_text(self, cell) -> str:
        segments: List[str] = []
        text_content = normalise_text(cell.get_text(" ", strip=True))
        if text_content:
            segments.append(text_content)

        link_segments: List[str] = []
        for anchor in cell.find_all("a", href=True):
            href = anchor.get("href")
            if not href:
                continue
            url_value = urljoin(BASE_URL, href)
            link_text = normalise_text(anchor.get_text(" ", strip=True))
            if link_text and link_text.lower() != url_value.lower():
                link_segments.append(f"{link_text} ({url_value})")
            else:
                link_segments.append(url_value)
        if link_segments:
            segments.append("; ".join(link_segments))

        return " | ".join(segment for segment in segments if segment)

    @staticmethod
    def _derive_label(fields: Dict[str, str]) -> Optional[str]:
        if not fields:
            return None
        preferred_keys = [
            "JP identification name",
            "Crude drug name",
            "Crude drug",
            "Title",
            "Name",
        ]
        for key in preferred_keys:
            value = fields.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
        for value in fields.values():
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
        return None

    def _download_photos(self, entry: JPIdentificationEntry) -> None:
        target_dir = self.photo_root / str(entry.jp_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        for photo in entry.photos:
            source_url = photo.full_size_url or photo.thumbnail_url
            if not source_url:
                continue
            file_name = photo.file_name or f"{photo.photo_id}.jpg"
            destination = target_dir / file_name
            if destination.exists():
                logger.debug("Skipping existing photo %s", destination)
                continue
            try:
                response = self.session.get(source_url, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:
                logging.warning(
                    "Failed to download photo %s for JP identification %s: %s",
                    source_url,
                    entry.jp_id,
                    exc,
                )
                continue
            destination.write_bytes(response.content)
            logger.info("Saved photo %s -> %s", source_url, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl MPDB JP identification entries and optionally download photos.",
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START_ID, help="First JP identification ID to crawl (inclusive).")
    parser.add_argument("--stop", type=int, default=DEFAULT_STOP_ID, help="Last JP identification ID to crawl (inclusive).")
    parser.add_argument("--limit", type=int, help="Maximum number of entries to collect.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay (seconds) between requests.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Language parameter appended to detail page requests.")
    parser.add_argument(
        "--download-photos",
        action="store_true",
        help="Download photos for each entry using the discovered image links.",
    )
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=Path("crawler/photos/jp_identification"),
        help="Directory where photos will be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file path to store the scraped entries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    crawler = JPIdentificationCrawler(
        lang=args.lang,
        delay=args.delay,
        timeout=args.timeout,
        download_photos=args.download_photos,
        photo_root=args.photo_dir,
    )

    try:
        entries = crawler.crawl(start=args.start, stop=args.stop, limit=args.limit)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    field_names = sorted({key for entry in entries for key in entry.fields})
    csv_columns = ["jp_id", "url", "label", *field_names, "photos", "memos"]

    def entry_to_row(entry: JPIdentificationEntry) -> Dict[str, str]:
        row: Dict[str, str] = {
            "jp_id": str(entry.jp_id),
            "url": entry.url,
            "label": entry.label or "",
        }
        for field in field_names:
            value = entry.fields.get(field)
            row[field] = value if value is not None else ""
        if entry.photos:
            row["photos"] = json.dumps([photo.to_dict() for photo in entry.photos], ensure_ascii=False)
        else:
            row["photos"] = ""
        row["memos"] = " | ".join(entry.memos)
        return row

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_columns)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry_to_row(entry))
        logger.info("Wrote %s entries to %s", len(entries), args.output)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=csv_columns)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry_to_row(entry))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
