#!/usr/bin/env python3
"""
Utility crawler for the Comprehensive Medicinal Plant Database (MPDB).

The script downloads list data for one or more categories and, optionally,
grabs the associated detail pages. Content is exported as JSON so that it can
be stored or processed further downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "http://mpdb.nibiohn.go.jp"
LIST_ENDPOINT = "/mpdb-bin/list_data.cgi"
SUPPORTED_CATEGORIES = {
    "plant": "Plant data list",
    "crudedrug": "Crude drug data list",
    "jplist": "JP crude drug list",
    "syohou": "Kampo formula data list",
    "compound": "Compound data list",
    "sample": "Sample data list",
}

CSV_FILE_ENCODING = "utf-8"
DEFAULT_HTML_ENCODING = "utf-8"
HTTP_ERROR_LOG_TEMPLATE = "crawler_http_errors_{category}.log"
CRUDEDRUG_JP_DATA_FIELDS = {
    "gene data",
    "identification",
    "identification (tlc)",
    "loss on drying",
    "total ash",
    "acid-insoluble ash",
    "extract content",
    "essential oil content",
    "purity",
    "other",
}

logger = logging.getLogger(__name__)
_HTTP_ERROR_LOGGERS: Dict[str, logging.Logger] = {}


def _normalise_log_category(category: Optional[str]) -> str:
    base = (category or "general").strip().lower()
    cleaned = re.sub(r"[^\w]+", "_", base)
    cleaned = cleaned.strip("_")
    return cleaned or "general"


def _get_http_logger(category: Optional[str]) -> logging.Logger:
    normalized = _normalise_log_category(category)
    cached = _HTTP_ERROR_LOGGERS.get(normalized)
    if cached:
        return cached
    logger_name = f"{__name__}.http.{normalized}"
    http_logger = logging.getLogger(logger_name)
    if not http_logger.handlers:
        log_path = Path(HTTP_ERROR_LOG_TEMPLATE.format(category=normalized))
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        http_logger.addHandler(handler)
    http_logger.setLevel(logging.ERROR)
    http_logger.propagate = False
    _HTTP_ERROR_LOGGERS[normalized] = http_logger
    return http_logger

_URL_PATTERN = re.compile(r"(https?://\S+|ftp://\S+|www\.\S+)", re.IGNORECASE)


def _is_resource_field(category: str, name: Optional[str]) -> bool:
    """Return True when the field represents plant resource data."""
    if category != "plant":
        return False
    return (name or "").strip().lower() == "resource data"


def normalise_text(value: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    return " ".join(value.split())


@dataclass
class CrawlResult:
    category: str
    columns: List[str]
    items: List[Dict]


class MPDBCrawler:
    """Simple HTML crawler specialised for mpdb.nibiohn.go.jp."""

    def __init__(
        self,
        lang: str = "en",
        delay: float = 0.5,
        timeout: float = 30.0,
        download_photos: bool = False,
        photo_root: Optional[Path] = None,
    ):
        self.lang = lang
        self.delay = delay
        self.timeout = timeout
        self.download_photos = download_photos
        self.photo_root = Path(photo_root) if photo_root else Path("photos")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "mpdb-crawler/1.0 (+https://github.com/)",
                "Accept-Language": lang,
                "Accept-Charset": DEFAULT_HTML_ENCODING,
            }
        )

    def crawl(
        self,
        categories: Iterable[str],
        include_details: bool = False,
        limit: Optional[int] = None,
    ) -> List[CrawlResult]:
        results: List[CrawlResult] = []
        for category in categories:
            logging.info("Crawling category '%s'", category)
            try:
                items, headers = self._crawl_category(category, include_details=include_details, limit=limit)
            except requests.RequestException as exc:
                logging.error("Skipping category '%s' due to HTTP error: %s", category, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logging.error("Skipping category '%s' due to unexpected error: %s", category, exc, exc_info=True)
                continue
            results.append(CrawlResult(category=category, columns=headers, items=items))
        return results

    def _fetch(self, url: str, dataType: str, params: Optional[Dict] = None, category: Optional[str] = None) -> str:
        """Fetch raw HTML, raising for HTTP errors."""
        logging.info("Fetching data from url: %s, data type: %s", url, dataType)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            self._log_http_error("timeout", url, exc, {"params": params, "data_type": dataType}, category=category)
            raise
        except requests.RequestException as exc:
            self._log_http_error("http_error", url, exc, {"params": params, "data_type": dataType}, category=category)
            raise
        encoding = response.encoding or response.apparent_encoding or DEFAULT_HTML_ENCODING
        if encoding.lower() == "iso-8859-1":
            encoding = DEFAULT_HTML_ENCODING
        try:
            return response.content.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            logging.debug("Failed to decode %s with %s; falling back to %s", url, encoding, DEFAULT_HTML_ENCODING)
            return response.content.decode(DEFAULT_HTML_ENCODING, errors="replace")

    def _crawl_category(
        self,
        category: str,
        include_details: bool = False,
        limit: Optional[int] = None,
    ) -> Tuple[List[Dict], List[str]]:
        """Return table entries and column order for a single category."""
        url = urljoin(BASE_URL, LIST_ENDPOINT)
        html = self._fetch(url, 'Category', params={"category": category, "lang": self.lang}, category=category)
        soup = BeautifulSoup(html, "html.parser")

        table = soup.find("table", class_="list")
        if table is None:
            raise RuntimeError(f"Unable to locate data table for category '{category}'.")

        table_headers = [normalise_text(th.get_text(" ", strip=True)) for th in table.select("thead tr th")]
        skip_column_indexes: Set[int] = set()
        filtered_headers: List[str] = []
        for idx, header in enumerate(table_headers):
            if _is_resource_field(category, header):
                skip_column_indexes.add(idx)
                continue
            filtered_headers.append(header)
        headers = filtered_headers if "ID" in filtered_headers else ["ID"] + filtered_headers
        items: List[Dict] = []
        tbody = table.find("tbody")
        if not tbody:
            logging.warning("No table body found for category '%s'", category)
            return items, headers

        for row_idx, tr in enumerate(tbody.find_all("tr"), start=1):
            cells = tr.find_all("td")
            if not cells:
                continue

            detail_data: Optional[Dict[str, object]] = None
            row_data: Dict[str, object] = {}
            detail_link = tr.find("a", href=True)
            detail_url = urljoin(BASE_URL, detail_link["href"]) if detail_link else None
            entry_id = None
            if detail_url:
                entry_id = self._extract_id(detail_url)
                if entry_id:
                    row_data["ID"] = entry_id

            for column_index, (header, cell) in enumerate(zip(table_headers, cells)):
                if column_index in skip_column_indexes:
                    continue
                value_text = normalise_text(cell.get_text(" ", strip=True))
                links = [
                    {"text": normalise_text(a.get_text(" ", strip=True)), "url": urljoin(BASE_URL, a.get("href", ""))}
                    for a in cell.find_all("a")
                    if a.get("href")
                ]
                if links:
                    row_data[header] = {"text": value_text or None, "links": links}
                else:
                    row_data[header] = value_text or None

            entry: Dict[str, object] = {"row_number": row_idx, "columns": row_data}
            if detail_url:
                entry["detail_url"] = detail_url
            if entry_id and "id" not in entry:
                entry["id"] = entry_id

            if include_details and detail_url:
                time.sleep(self.delay)
                detail_data = self._filter_detail_fields(category, self._fetch_detail(detail_url, category=category))
                entry["detail"] = detail_data

            if self.download_photos and detail_url:
                if detail_data is None:
                    time.sleep(self.delay)
                    detail_data = self._filter_detail_fields(category, self._fetch_detail(detail_url, category=category))
                    if include_details:
                        entry["detail"] = detail_data
                photos = detail_data.get("photos") if detail_data else None
                if photos:
                    identifier = self._derive_photo_identifier(category, row_data, detail_data)
                    if identifier:
                        self._download_photos_for_category(category, identifier, photos)

            items.append(entry)

            if limit and len(items) >= limit:
                logging.info("Hit limit (%s) for category '%s'", limit, category)
                break

            if (include_details and detail_url) or (self.download_photos and detail_url):
                continue  # delay already applied
            time.sleep(self.delay)

        return items, headers

    def _fetch_detail(self, detail_url: str, category: Optional[str] = None) -> Dict[str, object]:
        """Fetch and parse a detail page into structured rows."""
        try:
            html = self._fetch(detail_url, 'Detail', category=category)
        except requests.RequestException as exc:
            logging.warning("Skipping detail page %s due to HTTP error: %s", detail_url, exc)
            return {}
        soup = BeautifulSoup(html, "html.parser")
        detail_table = soup.find("table", class_="detail")
        if detail_table is None:
            logging.warning("Detail table not found at %s", detail_url)
            return {}

        rows: List[Dict[str, object]] = []
        photo_entries: List[Dict[str, object]] = []
        for tr in detail_table.find_all("tr", recursive=False):
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue

            # Rows where the page uses a single cell with colspan=2.
            if len(cells) == 1:
                cell = cells[0]
                text_value = normalise_text(cell.get_text(" ", strip=True))
                links = self._extract_links(cell)
                images = self._extract_images(cell)
                if images:
                    for image in images:
                        photo_entries.append({"source": "inline", **image})
                rows.append(
                    {
                        "label": None,
                        "value": text_value or None,
                        "links": links or None,
                        "images": images or None,
                    }
                )
                continue

            label = normalise_text(cells[0].get_text(" ", strip=True))
            value_cell = cells[1]
            value_text = normalise_text(value_cell.get_text(" ", strip=True))
            links = self._extract_links(value_cell)
            images = self._extract_images(value_cell)
            if images:
                for image in images:
                    photo_entries.append({"source": "inline", **image})

            if label.lower() == "photo library" and links:
                library_photos: List[Dict[str, object]] = []
                for link in links:
                    time.sleep(self.delay)
                    try:
                        library_photos.extend(self._fetch_photo_library(link["url"], category=category))
                    except requests.RequestException as exc:
                        logging.warning("Skipping photo library %s due to HTTP error: %s", link["url"], exc)
                        continue
                if library_photos:
                    for photo in library_photos:
                        photo_entries.append({"source": "library", **photo})

            rows.append(
                {
                    "label": label or None,
                    "value": value_text or None,
                    "links": links or None,
                    "images": images or None,
                }
            )

        detail: Dict[str, object] = {"rows": rows}
        if photo_entries:
            detail["photos"] = photo_entries
        return detail

    @staticmethod
    def _filter_detail_fields(category: str, detail: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        """Remove unwanted fields from detail payloads."""
        if not isinstance(detail, dict):
            return detail
        if category == "plant":
            rows = detail.get("rows")
            if isinstance(rows, list):
                detail["rows"] = [
                    row
                    for row in rows
                    if not (
                        isinstance(row, dict)
                        and _is_resource_field(category, row.get("label"))
                    )
                ]
        if category == "crudedrug":
            MPDBCrawler._flatten_crudedrug_jp_data(detail)
        return detail

    @staticmethod
    def _flatten_crudedrug_jp_data(detail: Dict[str, object]) -> None:
        """Flatten JP data rows so all entries land in a single field."""
        rows = detail.get("rows")
        if not isinstance(rows, list):
            return

        jp_row_index: Optional[int] = None
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            label = row.get("label")
            if isinstance(label, str) and label.strip().lower() == "jp data":
                jp_row_index = index
                break
        if jp_row_index is None:
            return

        jp_row = rows[jp_row_index]
        aggregated_texts = MPDBCrawler._collect_detail_texts(jp_row)
        aggregated_links = list(MPDBCrawler._extract_detail_links(jp_row))
        consumed_indexes: List[int] = []

        for index, row in enumerate(rows):
            if index == jp_row_index:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("label"):
                continue
            texts = MPDBCrawler._collect_detail_texts(row)
            if not texts:
                continue
            first_token = MPDBCrawler._normalise_detail_token(texts[0])
            if not first_token or first_token not in CRUDEDRUG_JP_DATA_FIELDS:
                continue
            consumed_indexes.append(index)
            aggregated_texts.extend(texts)
            aggregated_links.extend(MPDBCrawler._extract_detail_links(row))

        if not consumed_indexes:
            return

        combined_texts = MPDBCrawler._deduplicate_tokens(aggregated_texts)
        unique_links = MPDBCrawler._deduplicate_links(aggregated_links)

        if combined_texts:
            jp_row["value"] = {"text": "; ".join(combined_texts)}
        else:
            jp_row["value"] = None

        jp_row["links"] = unique_links
        if isinstance(jp_row["value"], dict):
            jp_row["value"]["links"] = unique_links or None

        for index in sorted(consumed_indexes, reverse=True):
            del rows[index]

    @staticmethod
    def _collect_detail_texts(row: Dict[str, object]) -> List[str]:
        texts: List[str] = []
        texts.extend(MPDBCrawler._value_to_strings(row.get("value")))
        for link in row.get("links") or []:
            if not isinstance(link, dict):
                continue
            text = normalise_text(link.get("text") or "")
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _extract_detail_links(row: Dict[str, object]) -> List[Dict[str, Optional[str]]]:
        links: List[Dict[str, Optional[str]]] = []
        for link in row.get("links") or []:
            if not isinstance(link, dict):
                continue
            text = normalise_text(link.get("text") or "")
            url = link.get("url")
            if text or url:
                links.append({"text": text or None, "url": url})
        return links

    @staticmethod
    def _normalise_detail_token(value: str) -> str:
        return normalise_text(value).lower()

    @staticmethod
    def _value_to_strings(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = normalise_text(value)
            return [text] if text else []
        if isinstance(value, dict):
            text = normalise_text(value.get("text") or "")
            return [text] if text else []
        if isinstance(value, list):
            collected: List[str] = []
            for item in value:
                collected.extend(MPDBCrawler._value_to_strings(item))
            return collected
        text = normalise_text(str(value))
        return [text] if text else []

    @staticmethod
    def _deduplicate_tokens(tokens: Iterable[str]) -> List[str]:
        ordered: List[str] = []
        seen: Set[str] = set()
        for token in tokens:
            normalized = normalise_text(token)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _deduplicate_links(links: Iterable[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
        ordered: List[Dict[str, Optional[str]]] = []
        seen: Set[Tuple[Optional[str], Optional[str]]] = set()
        for link in links:
            if not isinstance(link, dict):
                continue
            text = normalise_text(link.get("text") or "")
            url = link.get("url")
            key = (text or None, url)
            if key in seen:
                continue
            seen.add(key)
            ordered.append({"text": text or None, "url": url})
        return ordered

    @staticmethod
    def _extract_links(cell) -> List[Dict[str, str]]:
        """Extract anchor data from a table cell."""
        links = []
        for anchor in cell.find_all("a"):
            href = anchor.get("href")
            if not href:
                continue
            links.append(
                {
                    "text": normalise_text(anchor.get_text(" ", strip=True)),
                    "url": urljoin(BASE_URL, href),
                }
            )
        return links

    @staticmethod
    def _extract_images(cell) -> List[Dict[str, Optional[str]]]:
        """Extract image metadata from a table cell."""
        images: List[Dict[str, Optional[str]]] = []
        for img in cell.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            if "plusminus" in src.lower():
                continue
            if "contents_root" not in src.lower() and "get_image.cgi" not in src.lower():
                continue
            thumb_url = urljoin(BASE_URL, src) if src else None
            alt_text = normalise_text(img.get("alt") or "") or None
            anchor = img.find_parent("a")
            full_url = urljoin(BASE_URL, anchor.get("href")) if anchor and anchor.get("href") else None
            images.append(
                {
                    "thumbnail_url": thumb_url,
                    "full_size_url": full_url,
                    "alt": alt_text,
                }
            )
        return images

    def _fetch_photo_library(self, photo_page_url: str, category: Optional[str] = None) -> List[Dict[str, object]]:
        """Fetch photo library entries from the dedicated plant photo page."""
        html = self._fetch(photo_page_url, 'PhotoLibrary', category=category)
        soup = BeautifulSoup(html, "html.parser")
        photo_main = soup.find("table", class_="photo_main")
        if photo_main is None:
            return []

        photos: List[Dict[str, object]] = []
        for photo_table in photo_main.find_all("table", class_="photo"):
            anchor = photo_table.find("a", href=True)
            full_url = urljoin(BASE_URL, anchor["href"]) if anchor else None
            img_tag = photo_table.find("img", src=True)
            thumb_url = urljoin(BASE_URL, img_tag["src"]) if img_tag else None

            text_lines: List[str] = []
            for tr in photo_table.find_all("tr")[1:]:
                text = normalise_text(tr.get_text(" ", strip=True))
                if text:
                    text_lines.append(text)

            file_name = text_lines[0] if text_lines else None
            extra_text = text_lines[1:] if len(text_lines) > 1 else None

            photos.append(
                {
                    "full_size_url": full_url,
                    "thumbnail_url": thumb_url,
                    "file_name": file_name,
                    "notes": extra_text,
                }
            )

        return photos

    @staticmethod
    def _extract_plant_name(columns: Dict[str, object]) -> Optional[str]:
        """Extract plant latin name text from row columns."""
        plant_entry = columns.get("Plant latin name")
        if isinstance(plant_entry, dict):
            return plant_entry.get("text")
        if isinstance(plant_entry, str):
            return plant_entry
        return None

    @staticmethod
    def _value_to_text(value: object) -> Optional[str]:
        """Convert a stored cell value back into plain text."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get("text") or None
        return str(value)

    def _find_detail_value(self, detail: Optional[Dict[str, object]], label: str) -> Optional[str]:
        """Return the text associated with a labelled detail row."""
        if not detail:
            return None
        rows = detail.get("rows", [])
        for row in rows:
            row_label = row.get("label")
            if not row_label:
                continue
            if row_label.strip().lower() == label.strip().lower():
                return row.get("value")
        return None

    def _derive_photo_identifier(
        self,
        category: str,
        row_data: Dict[str, object],
        detail: Optional[Dict[str, object]],
    ) -> Optional[str]:
        """Determine a filesystem-safe identifier for photo downloads."""
        if category == "plant":
            name = self._extract_plant_name(row_data)
            return name or self._value_to_text(row_data.get("ID"))

        if category == "sample":
            serial_candidates = [
                self._find_detail_value(detail, "Serial Number"),
                self._value_to_text(row_data.get("Serial Number")),
            ]
            serial = next((value for value in serial_candidates if value), None)
            return serial or self._value_to_text(row_data.get("ID"))

        if category == "compound":
            compound_name = self._value_to_text(row_data.get("Compound name"))
            if not compound_name:
                compound_name = self._find_detail_value(detail, "Compound name")
            return compound_name or self._value_to_text(row_data.get("ID"))

        return self._value_to_text(row_data.get("ID"))

    @staticmethod
    def _extract_id(url: str) -> Optional[str]:
        """Extract the 'id' query parameter from a URL."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        values = params.get("id")
        if not values:
            return None
        return values[0]

    @staticmethod
    def _safe_segment(value: str, fallback: str = "unknown") -> str:
        """Return a filesystem-safe segment derived from the provided value."""
        if not value:
            return fallback
        cleaned = re.sub(r"[^\w\s.-]", "_", value.strip())
        cleaned = re.sub(r"[\s]+", "_", cleaned)
        return cleaned or fallback

    def _download_photos_for_category(self, category: str, identifier: str, photos: List[Dict[str, object]]) -> None:
        """Download photos into the configured directory for a specific category."""
        category_dir = {
            "plant": "plants",
            "sample": "samples",
            "compound": "compounds",
        }.get(category, category)
        target_dir = self.photo_root / category_dir / self._safe_segment(identifier)
        target_dir.mkdir(parents=True, exist_ok=True)

        for index, photo in enumerate(photos, start=1):
            url = photo.get("full_size_url") or photo.get("thumbnail_url")
            if not url:
                continue

            filename = photo.get("file_name")
            if not filename:
                parsed = urlparse(url)
                filename = Path(parsed.path).name or f"photo_{index}.jpg"
            ext = Path(filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
                filename = f"photo_{index}.jpg"
            safe_name = self._safe_segment(filename, fallback=f"photo_{index}.jpg")
            destination = target_dir / safe_name

            if destination.exists():
                logging.debug("Photo already exists, skipping: %s", destination)
                continue

            time.sleep(self.delay)
            self._download_file(url, destination, category=category)

    def _download_file(self, url: str, destination: Path, category: Optional[str] = None) -> None:
        """Download binary content to a destination path."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            self._log_http_error("timeout_download", url, exc, {"destination": str(destination)}, category=category)
            logging.warning("Failed to download %s: %s", url, exc)
            return
        except requests.RequestException as exc:
            self._log_http_error("http_error_download", url, exc, {"destination": str(destination)}, category=category)
            logging.warning("Failed to download %s: %s", url, exc)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            handle.write(response.content)
        logging.info("Downloaded photo -> %s", destination)

    def _log_http_error(
        self,
        stage: str,
        url: str,
        exc: BaseException,
        extra: Optional[Dict[str, object]] = None,
        category: Optional[str] = None,
    ) -> None:
        """Persist HTTP issues so transient failures can be inspected later."""
        http_logger = _get_http_logger(category)
        http_logger.error(url)

        context_parts = [f"stage={stage}", f"url={url}", f"category={_normalise_log_category(category)}"]
        if extra:
            for key, value in extra.items():
                context_parts.append(f"{key}={value!r}")
        logger.debug("HTTP failure recorded: %s :: %s", " ".join(context_parts), exc, exc_info=True)


def _strip_urls(text: str) -> str:
    """Remove URL-like substrings and collapse whitespace."""
    if not text:
        return ""
    cleaned = _URL_PATTERN.sub("", text)
    return normalise_text(cleaned)


def cell_to_string(value: Any) -> str:
    """Convert a column cell value into a plain string for CSV export."""
    if value is None:
        return ""
    if isinstance(value, dict):
        text = _strip_urls(value.get("text"))
        if text:
            return text
        link_entries = value.get("links") or []
        link_texts: List[str] = []
        for link in link_entries:
            if not isinstance(link, dict):
                continue
            link_text = _strip_urls(link.get("text"))
            if link_text:
                link_texts.append(link_text)
        if link_texts:
            return ", ".join(link_texts)
        alt = _strip_urls(value.get("alt"))
        if alt:
            return alt
        description = _strip_urls(value.get("description"))
        if description:
            return description
        return ""
    if isinstance(value, list):
        return ", ".join(cell_to_string(item) for item in value)
    return _strip_urls(str(value))


def flatten_detail_row(row: Dict[str, object]) -> str:
    """Turn a detail row entry into a single string."""
    if not isinstance(row, dict):
        return cell_to_string(row)
    parts: List[str] = []

    value = row.get("value")
    value_str = ""
    if value:
        value_str = cell_to_string(value)
        if value_str:
            parts.append(value_str)

    link_entries = row.get("links") or []
    link_strings: List[str] = []
    for link in link_entries:
        if not isinstance(link, dict):
            continue
        text = _strip_urls(link.get("text"))
        if text:
            link_strings.append(text)
    if link_strings:
        cleaned_links: List[str] = []
        seen_links: Set[str] = set()
        value_tokens: Set[str] = {
            normalise_text(token).lower()
            for token in re.split(r"[;|]", value_str)
            if token.strip()
        } if value_str else set()
        for link_text in link_strings:
            normalized = normalise_text(link_text).lower()
            if not normalized or normalized in value_tokens or normalized in seen_links:
                continue
            seen_links.add(normalized)
            cleaned_links.append(link_text)
        if cleaned_links:
            parts.append("; ".join(cleaned_links))

    image_entries = row.get("images") or []
    image_strings: List[str] = []
    for image in image_entries:
        if not isinstance(image, dict):
            continue
        fragments: List[str] = []
        alt = _strip_urls(image.get("alt"))
        if alt:
            fragments.append(alt)
        if fragments:
            image_strings.append(" ".join(fragments))
    if image_strings:
        parts.append("; ".join(image_strings))

    return " | ".join(part for part in parts if part)


def flatten_photo_entries(photos: List[Dict[str, object]]) -> str:
    """Flatten detail photo metadata into a CSV-friendly string."""
    photo_strings: List[str] = []
    for photo in photos:
        if not isinstance(photo, dict):
            continue
        fragments: List[str] = []
        file_name = cell_to_string(photo.get("file_name"))
        if file_name:
            fragments.append(file_name)
        notes = photo.get("notes")
        if isinstance(notes, list):
            note_text = "; ".join(cell_to_string(note) for note in notes if note)
        else:
            note_text = cell_to_string(notes)
        if note_text:
            fragments.append(note_text)
        if fragments:
            photo_strings.append(" | ".join(fragments))
    return "; ".join(photo_strings)


def _should_skip_column(column_name: str, category: str) -> bool:
    """Return True when a column should be omitted from CSV export."""
    normalized = (column_name or "").strip().lower()
    if not normalized:
        return False
    if category == "plant" and normalized == "resource data":
        return True
    return "url" in normalized


def write_csv_result(result: CrawlResult, handle: TextIO) -> None:
    """Write a CrawlResult into a CSV file handle."""
    base_columns = [column for column in result.columns if not _should_skip_column(column, result.category)]
    base_fieldnames = ["row_number"] + base_columns

    detail_columns: List[str] = []
    seen_detail_labels: set[str] = set()
    max_unlabelled_rows = 0
    include_photo_column = False

    for item in result.items:
        detail = item.get("detail") if isinstance(item, dict) else None
        if not isinstance(detail, dict):
            continue
        rows = detail.get("rows") or []
        unlabeled_count = 0
        for row in rows:
            label = row.get("label")
            if label:
                label_str = cell_to_string(label)
                if label_str and not _should_skip_column(label_str, result.category) and label_str not in seen_detail_labels:
                    detail_columns.append(label_str)
                    seen_detail_labels.add(label_str)
            else:
                unlabeled_count += 1
        if unlabeled_count > max_unlabelled_rows:
            max_unlabelled_rows = unlabeled_count
        if detail.get("photos"):
            include_photo_column = True

    unlabelled_columns = [f"Unlabelled detail {index}" for index in range(1, max_unlabelled_rows + 1)]

    fieldnames = base_fieldnames.copy()
    fieldnames.extend(detail_columns)
    fieldnames.extend(unlabelled_columns)
    if include_photo_column:
        fieldnames.append("Detail photos")

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for item in result.items:
        row: Dict[str, str] = {"row_number": str(item.get("row_number", ""))}
        columns = item.get("columns", {})
        for column in base_columns:
            row[column] = cell_to_string(columns.get(column))

        detail = item.get("detail") if isinstance(item, dict) else None
        labelled_values: Dict[str, List[str]] = {}
        unlabelled_values: List[str] = []
        if isinstance(detail, dict):
            rows = detail.get("rows") or []
            for detail_row in rows:
                flattened = flatten_detail_row(detail_row)
                if not flattened:
                    continue
                label = detail_row.get("label")
                if label:
                    label_str = cell_to_string(label)
                    if label_str and not _should_skip_column(label_str, result.category):
                        labelled_values.setdefault(label_str, []).append(flattened)
                    continue
                unlabelled_values.append(flattened)

            if include_photo_column and detail.get("photos"):
                row["Detail photos"] = flatten_photo_entries(detail.get("photos") or [])

        for label in detail_columns:
            values = labelled_values.get(label, [])
            row[label] = "; ".join(values)

        for index, column_name in enumerate(unlabelled_columns):
            row[column_name] = unlabelled_values[index] if index < len(unlabelled_values) else ""
        if include_photo_column and "Detail photos" not in row:
            row["Detail photos"] = ""

        writer.writerow(row)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl mpdb.nibiohn.go.jp list and detail pages.")
    parser.add_argument(
        "-c",
        "--category",
        action="append",
        choices=sorted(SUPPORTED_CATEGORIES.keys()),
        help="Category to crawl (repeat for multiple). Defaults to 'plant'.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Follow detail links for each row and include parsed data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of rows pulled from each category (useful for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (seconds) between HTTP requests (default: 0.5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--download-photos",
        action="store_true",
        help="Download plant photos into 'photos/plants/<plant_latin_name>/' directories.",
    )
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=Path("photos"),
        help="Root output directory for downloaded photos (default: ./photos).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format for scraped data (default: json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output JSON file. Defaults to stdout.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    categories = args.category or ["plant"]
    if args.download_photos and not args.include_details:
        logging.info(
            "Photo download enabled. Detail pages will be fetched for plants even though they are not included in the output."
        )
    crawler = MPDBCrawler(
        lang="en",
        delay=max(0.0, args.delay),
        timeout=args.timeout,
        download_photos=args.download_photos,
        photo_root=args.photo_dir,
    )

    results: List[CrawlResult] = []
    try:
        results = crawler.crawl(categories=categories, include_details=args.include_details, limit=args.limit)
    except requests.HTTPError as exc:
        logging.error("HTTP error encountered while crawling: %s", exc)
        logging.info("Continuing without data for the failing request.")
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Unexpected error: %s", exc, exc_info=args.verbose)
        return 1

    if args.format == "json":
        serialisable = [
            {"category": result.category, "items": result.items} for result in results
        ]
        payload = json.dumps(serialisable, ensure_ascii=False, indent=2)

        if args.output:
            args.output.write_text(payload, encoding="utf-8")
            logging.info("Wrote %d categories to %s", len(serialisable), args.output)
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            sys.stdout.write(payload)
            sys.stdout.write("\n")
        return 0

    # CSV output path handling
    if args.format == "csv":
        if not args.output and len(results) > 1:
            logging.error("CSV output requires --output when multiple categories are requested.")
            return 1

        if not args.output:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", newline="")
            write_csv_result(results[0], sys.stdout)
            return 0

        output_path = args.output
        if len(results) == 1:
            result = results[0]
            if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
                output_path.mkdir(parents=True, exist_ok=True)
                file_path = output_path / f"{result.category}.csv"
            else:
                file_path = output_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding=CSV_FILE_ENCODING, newline="") as handle:
                write_csv_result(result, handle)
            logging.info("Wrote category '%s' to %s", result.category, file_path)
            return 0

        # multiple categories
        if output_path.suffix:
            logging.error("When exporting multiple categories to CSV, --output must be a directory.")
            return 1
        output_path.mkdir(parents=True, exist_ok=True)
        for result in results:
            file_path = output_path / f"{result.category}.csv"
            with file_path.open("w", encoding=CSV_FILE_ENCODING, newline="") as handle:
                write_csv_result(result, handle)
            logging.info("Wrote category '%s' to %s", result.category, file_path)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
