from __future__ import annotations

import logging
import time
import urllib.parse
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning
from urllib3.exceptions import NotOpenSSLWarning

if TYPE_CHECKING:  # pragma: no cover
    from mpdb_crawler import MPDBCrawler

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

logger = logging.getLogger(__name__)

BASE_URL = "http://mpdb.nibiohn.go.jp"
PLANT_LIST_URL = f"{BASE_URL}/mpdb-bin/list_data.cgi?category=plant"

EXPECTED_COLUMNS: List[str] = [
    "ID",
    "Plant latin name",
    "Family name",
    "Common name",
    "Crude drug latin name",
    "Cultured tissue and efficient propagation",
    "Plant culture and efficient production method",
    "Sakuyo hyohon",
    "Class of Coldness",
    "Class of Warmness",
    "Light condition",
    "Preference of Soil",
    "Requirement of Shading",
]


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            )
        }
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _absolute_url(href: str) -> str:
    return urllib.parse.urljoin(BASE_URL, href)


def _extract_id_from_url(url: str) -> Optional[str]:
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        ids = params.get("id")
        if ids:
            return ids[0]
    except Exception:
        return None
    return None


def _text_or_empty(node) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def _extract_links(node) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    if node is None:
        return links
    for link in node.find_all("a", href=True):
        href = link.get("href", "")
        absolute = _absolute_url(href)
        links.append(
            {
                "text": _text_or_empty(link),
                "url": absolute,
                "id": _extract_id_from_url(absolute),
            }
        )
    return links


def _extract_photo_entries(node) -> List[Dict[str, Optional[str]]]:
    """Extract photo metadata (full-size and thumbnail URLs) from a cell."""
    entries: List[Dict[str, Optional[str]]] = []
    seen_urls: Set[str] = set()
    if node is None:
        return entries

    for anchor in node.find_all("a", href=True):
        full_url = _absolute_url(anchor.get("href", ""))
        thumb_url: Optional[str] = None
        img = anchor.find("img", src=True)
        if img and img.get("src"):
            thumb_url = _absolute_url(img["src"])
        alt_text = _text_or_empty(img) if img else _text_or_empty(anchor)

        file_name = Path(urllib.parse.urlparse(full_url).path).name
        if not file_name and thumb_url:
            file_name = Path(urllib.parse.urlparse(thumb_url).path).name
        entry = {
            "full_size_url": full_url or None,
            "thumbnail_url": thumb_url or None,
            "file_name": file_name or None,
            "alt": alt_text or None,
        }
        key = full_url or thumb_url
        if key and key not in seen_urls:
            entries.append(entry)
            seen_urls.add(key)
        if thumb_url:
            seen_urls.add(thumb_url)

    for img in node.find_all("img", src=True):
        thumb_url = _absolute_url(img["src"])
        if thumb_url in seen_urls:
            continue
        alt_text = _text_or_empty(img)
        file_name = Path(urllib.parse.urlparse(thumb_url).path).name
        entry = {
            "full_size_url": thumb_url,
            "thumbnail_url": thumb_url,
            "file_name": file_name or None,
            "alt": alt_text or None,
        }
        entries.append(entry)
        seen_urls.add(thumb_url)

    return entries


def _parse_growing_data(detail_table) -> Dict[str, str]:
    result = {
        "Class of Coldness": "",
        "Class of Warmness": "",
        "Light condition": "",
        "Preference of Soil": "",
        "Requirement of Shading": "",
    }

    label_row = None
    for tr in detail_table.find_all("tr", recursive=False):
        td = tr.find("td", class_="label_17")
        if td and "Growing data" in _text_or_empty(td):
            label_row = tr
            break

    if label_row is None:
        return result

    sibling = label_row.find_next_sibling("tr")
    if sibling is None:
        return result

    inner_cell = sibling.find("div", class_="cell")
    if inner_cell is None:
        return result

    inner_table = inner_cell.find("table")
    if inner_table is None:
        return result

    for tr in inner_table.find_all("tr"):
        tds = tr.find_all("td")
        i = 0
        while i < len(tds):
            label = _text_or_empty(tds[i])
            value = ""
            if i + 1 < len(tds):
                value = _text_or_empty(tds[i + 1])

            normalized_label = label.lower().strip()
            if normalized_label in {"class of coldness"}:
                result["Class of Coldness"] = value
            elif normalized_label in {"class of warmness"}:
                result["Class of Warmness"] = value
            elif normalized_label in {"light condition"}:
                result["Light condition"] = value
            elif normalized_label in {"preference of soil"}:
                result["Preference of Soil"] = value
            elif normalized_label in {"requirement of shading"}:
                result["Requirement of Shading"] = value

            i += 2

    return result


def _parse_html(content: bytes) -> BeautifulSoup:
    """Parse HTML content using preferred parser, falling back to built-in."""
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(content, parser)
        except FeatureNotFound:
            continue
    raise FeatureNotFound("No supported HTML parser (lxml or html.parser) is available.")


def fetch_plant_detail(url: str, session: Optional[requests.Session] = None) -> Dict[str, str]:
    sess = session or _create_session()
    logger.debug("Fetching plant detail page: %s", url)
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()

    soup = _parse_html(resp.content)
    detail_table = soup.find("table", class_="detail")
    if detail_table is None:
        return {}

    mapping = {
        "latin name": "Plant latin name",
        "family name": "Family name",
        "common name": "Common name",
        "crude drug": "Crude drug latin name",
        "cultured tissue and efficient propagation": "Cultured tissue and efficient propagation",
        "plant culture and efficient production method": "Plant culture and efficient production method",
        "sakuyo hyohon": "Sakuyo hyohon",
    }

    data: Dict[str, str] = {
        "ID": _extract_id_from_url(url) or "",
        "Plant latin name": "",
        "Family name": "",
        "Common name": "",
        "Crude drug latin name": "",
        "Cultured tissue and efficient propagation": "",
        "Plant culture and efficient production method": "",
        "Sakuyo hyohon": "",
        "Class of Coldness": "",
        "Class of Warmness": "",
        "Light condition": "",
        "Preference of Soil": "",
        "Requirement of Shading": "",
    }

    tissue_links: List[Dict[str, str]] = []
    cultivation_links: List[Dict[str, str]] = []

    for tr in detail_table.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        raw_label = _text_or_empty(tds[0])
        normalized_label = " ".join(raw_label.split()).lower()
        if normalized_label in mapping and len(tds) >= 2:
            key = mapping[normalized_label]
            value_cell = tds[1]
            extracted_links = _extract_links(value_cell)
            if extracted_links:
                if normalized_label == "cultured tissue and efficient propagation":
                    for link_info in extracted_links:
                        tissue_links.append(
                            {
                                "plant_id": data["ID"],
                                "plant_latin_name": data["Plant latin name"],
                                "link_text": _normalise_text(link_info.get("text") or ""),
                                "url": link_info.get("url") or "",
                                "tissue_id": link_info.get("id"),
                            }
                        )
                elif normalized_label == "plant culture and efficient production method":
                    for link_info in extracted_links:
                        cultivation_links.append(
                            {
                                "plant_id": data["ID"],
                                "plant_latin_name": data["Plant latin name"],
                                "link_text": _normalise_text(link_info.get("text") or ""),
                                "url": link_info.get("url") or "",
                                "cultivation_id": link_info.get("id"),
                            }
                        )
                values = [_normalise_text(link.get("text") or link.get("url") or "") for link in extracted_links]
                value = ", ".join(filter(None, values))
            else:
                value = _text_or_empty(value_cell)
            data[key] = value

    growing = _parse_growing_data(detail_table)
    data.update(growing)

    if tissue_links:
        data["_tissue_culture_links"] = tissue_links
    if cultivation_links:
        data["_cultivation_links"] = cultivation_links

    return data


def fetch_tissue_culture_literature(
    url: str,
    session: Optional[requests.Session] = None,
) -> Dict[str, str]:
    sess = session or _create_session()
    logger.debug("Fetching tissue culture literature page: %s", url)
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()

    soup = _parse_html(resp.content)
    detail_table = soup.find("table", class_="detail")
    if detail_table is None:
        return {"ID": _extract_id_from_url(url) or ""}

    data: Dict[str, str] = {}
    for tr in detail_table.find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        label = _text_or_empty(cells[0])
        if not label:
            continue
        value_cell = cells[1]
        link_entries = _extract_links(value_cell)
        if link_entries:
            value_parts = [_normalise_text(entry.get("text") or entry.get("url") or "") for entry in link_entries]
            value = "; ".join(filter(None, value_parts))
        else:
            value = _text_or_empty(value_cell)
        data[_normalise_text(label)] = _normalise_text(value)

    detail_id = data.get("ID") or _extract_id_from_url(url)
    if detail_id:
        data["ID"] = detail_id
    return data


def fetch_cultivation_data(
    url: str,
    session: Optional[requests.Session] = None,
) -> Dict[str, str]:
    sess = session or _create_session()
    logger.debug("Fetching cultivation data page: %s", url)
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()

    soup = _parse_html(resp.content)
    detail_table = soup.find("table", class_="detail")
    if detail_table is None:
        return {"ID": _extract_id_from_url(url) or ""}

    data: Dict[str, str] = {}
    photos: List[Dict[str, Optional[str]]] = []
    tr_list = list(detail_table.find_all("tr", recursive=False))
    i = 0
    while i < len(tr_list):
        tr = tr_list[i]
        cells = tr.find_all("td", recursive=False)
        
        # Handle rows with colspan=2 that contain section headers
        if len(cells) == 1:
            label = _text_or_empty(cells[0])
            normalized_label = _normalise_text(label).lower()
            # Check if this is the "Cultivation Photo" section header
            if "photo" in normalized_label and "cultivation" in normalized_label:
                # Look at the next row for the content (which has colspan=2)
                if i + 1 < len(tr_list):
                    next_tr = tr_list[i + 1]
                    next_cells = next_tr.find_all("td", recursive=False)
                    # Find the cell div that contains nested tables
                    content_cell = None
                    for cell in next_cells:
                        cell_div = cell.find("div", class_="cell")
                        if cell_div:
                            content_cell = cell_div
                            break
                    if content_cell is None:
                        # Fallback: use the first cell of the next row
                        if next_cells:
                            content_cell = next_cells[0]
                    
                    if content_cell:
                        photo_entries = _extract_photo_entries(content_cell)
                        if photo_entries:
                            photos.extend(photo_entries)
                            value_display = "; ".join(
                                entry.get("file_name")
                                or Path(urllib.parse.urlparse(entry.get("full_size_url") or "").path).name
                                or entry.get("full_size_url")
                                or ""
                                for entry in photo_entries
                            )
                        else:
                            value_display = _text_or_empty(content_cell)
                        data[_normalise_text(label)] = _normalise_text(value_display)
                        i += 2  # Skip both the header row and the content row
                        continue
        
        if len(cells) < 2:
            i += 1
            continue
        
        label = _text_or_empty(cells[0])
        if not label:
            i += 1
            continue
        value_cell = cells[1]
        normalized_label = _normalise_text(label).lower()
        if "photo" in normalized_label and "cultivation" in normalized_label:
            photo_entries = _extract_photo_entries(value_cell)
            if photo_entries:
                photos.extend(photo_entries)
                value_display = "; ".join(
                    entry.get("file_name")
                    or Path(urllib.parse.urlparse(entry.get("full_size_url") or "").path).name
                    or entry.get("full_size_url")
                    or ""
                    for entry in photo_entries
                )
            else:
                value_display = _text_or_empty(value_cell)
            data[_normalise_text(label)] = _normalise_text(value_display)
            i += 1
            continue

        link_entries = _extract_links(value_cell)
        if link_entries:
            value_parts = [_normalise_text(entry.get("text") or entry.get("url") or "") for entry in link_entries]
            value = "; ".join(filter(None, value_parts))
        else:
            value = _text_or_empty(value_cell)
        data[_normalise_text(label)] = _normalise_text(value)
        i += 1

    detail_id = data.get("ID") or _extract_id_from_url(url)
    if detail_id:
        data["ID"] = detail_id
    if photos:
        data["_photos"] = photos
    return data


def iter_plant_entry_urls(session: Optional[requests.Session] = None) -> Iterable[str]:
    sess = session or _create_session()
    logger.debug("Fetching plant list page: %s", PLANT_LIST_URL)
    resp = sess.get(PLANT_LIST_URL, timeout=30)
    resp.raise_for_status()
    soup = _parse_html(resp.content)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "view_plant_data.cgi" in href and "category=plant" not in href:
            parsed = urllib.parse.urlparse(_absolute_url(href))
            qs = urllib.parse.parse_qs(parsed.query)
            qs["lang"] = ["en"]
            new_query = urllib.parse.urlencode({k: v[0] for k, v in qs.items()})
            normalized = parsed._replace(query=new_query).geturl()
            yield normalized


def crawl_plants(
    limit: Optional[int] = None,
    delay_seconds: float = 0.5,
    session: Optional[requests.Session] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    session = session or _create_session()
    plant_results: List[Dict[str, str]] = []
    tissue_results: List[Dict[str, str]] = []
    cultivation_results: List[Dict[str, str]] = []

    count = 0
    for url in iter_plant_entry_urls(session=session):
        try:
            data = fetch_plant_detail(url, session=session)
            if data:
                tissue_links = data.pop("_tissue_culture_links", [])
                cultivation_links = data.pop("_cultivation_links", [])
                plant_results.append(data)
                if tissue_links:
                    for link_info in tissue_links:
                        detail_url = link_info.get("url")
                        if not detail_url:
                            continue
                        logger.debug(
                            "Fetching tissue culture detail for plant %s (%s): %s",
                            data.get("ID"),
                            data.get("Plant latin name"),
                            detail_url,
                        )
                        try:
                            tissue_detail = fetch_tissue_culture_literature(detail_url, session=session)
                        except Exception:
                            tissue_detail = {}
                        record: Dict[str, str] = {
                            "Plant ID": data.get("ID", ""),
                            "Plant latin name": data.get("Plant latin name", ""),
                            "Tissue literature name": link_info.get("link_text") or "",
                            "Tissue literature ID": link_info.get("tissue_id") or "",
                            "Tissue detail URL": detail_url,
                        }
                        if tissue_detail:
                            detail_id = tissue_detail.pop("ID", "")
                            if detail_id and not record["Tissue literature ID"]:
                                record["Tissue literature ID"] = detail_id
                            if detail_id:
                                record["Detail page ID"] = detail_id
                            for key, value in tissue_detail.items():
                                record[key] = value
                        cleaned_record: Dict[str, str] = {}
                        for key, value in record.items():
                            if value is None:
                                cleaned_record[key] = ""
                            elif isinstance(value, str):
                                cleaned_record[key] = _normalise_text(value)
                            else:
                                cleaned_record[key] = _normalise_text(str(value))
                        tissue_results.append(cleaned_record)
                if cultivation_links:
                    for link_info in cultivation_links:
                        detail_url = link_info.get("url")
                        if not detail_url:
                            continue
                        logger.debug(
                            "Fetching cultivation detail for plant %s (%s): %s",
                            data.get("ID"),
                            data.get("Plant latin name"),
                            detail_url,
                        )
                        try:
                            cultivation_detail = fetch_cultivation_data(detail_url, session=session)
                        except Exception:
                            cultivation_detail = {}
                        record: Dict[str, str] = {
                            "Plant ID": data.get("ID", ""),
                            "Plant latin name": data.get("Plant latin name", ""),
                            "Cultivation entry name": link_info.get("link_text") or "",
                            "Cultivation entry ID": link_info.get("cultivation_id") or "",
                            "Cultivation detail URL": detail_url,
                        }
                        photo_entries: List[Dict[str, Optional[str]]] = []
                        if cultivation_detail:
                            photo_entries = cultivation_detail.pop("_photos", [])
                            detail_id = cultivation_detail.pop("ID", "")
                            if detail_id and not record["Cultivation entry ID"]:
                                record["Cultivation entry ID"] = detail_id
                            if detail_id:
                                record["Detail page ID"] = detail_id
                            for key, value in cultivation_detail.items():
                                record[key] = value
                        if photo_entries:
                            record["Cultivation photo count"] = str(len(photo_entries))
                            file_names: List[str] = []
                            for entry in photo_entries:
                                name = entry.get("file_name")
                                if not name:
                                    candidate_url = entry.get("full_size_url") or entry.get("thumbnail_url") or ""
                                    name = Path(urllib.parse.urlparse(candidate_url).path).name
                                if name:
                                    file_names.append(name)
                            if file_names:
                                record["Cultivation photo files"] = ", ".join(file_names)
                            record["_photos"] = photo_entries
                        cleaned_record: Dict[str, str] = {}
                        for key, value in record.items():
                            if key == "_photos":
                                cleaned_record[key] = value
                                continue
                            if value is None:
                                cleaned_record[key] = ""
                            elif isinstance(value, str):
                                cleaned_record[key] = _normalise_text(value)
                            else:
                                cleaned_record[key] = _normalise_text(str(value))
                        cultivation_results.append(cleaned_record)
        except Exception:
            pass

        count += 1
        if limit is not None and count >= limit:
            break
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return plant_results, tissue_results, cultivation_results


def _normalise_text(value: str) -> str:
    """Collapse consecutive whitespace and strip the ends."""
    return " ".join(value.split())


def _value_to_string(value: Any) -> Optional[str]:
    """Convert crawler cell values into plain text."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = _normalise_text(value)
        return cleaned or None
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            cleaned = _normalise_text(text)
            if cleaned:
                return cleaned
        link_entries = value.get("links")
        if isinstance(link_entries, list):
            link_texts: List[str] = []
            for entry in link_entries:
                if not isinstance(entry, dict):
                    continue
                link_text = entry.get("text")
                if isinstance(link_text, str):
                    cleaned_link = _normalise_text(link_text)
                    if cleaned_link:
                        link_texts.append(cleaned_link)
            if link_texts:
                return ", ".join(link_texts)
        alt_text = value.get("alt")
        if isinstance(alt_text, str):
            cleaned_alt = _normalise_text(alt_text)
            if cleaned_alt:
                return cleaned_alt
        description = value.get("description")
        if isinstance(description, str):
            cleaned_description = _normalise_text(description)
            if cleaned_description:
                return cleaned_description
        return None
    if isinstance(value, list):
        collected: List[str] = []
        for item in value:
            converted = _value_to_string(item)
            if converted:
                collected.append(converted)
        if collected:
            return ", ".join(collected)
        return None
    cleaned_generic = _normalise_text(str(value))
    return cleaned_generic or None


def _build_tissue_dataset(tissue_entries: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    base_order = [
        "Plant ID",
        "Plant latin name",
        "Tissue literature ID",
        "Tissue literature name",
        "Tissue detail URL",
        "Detail page ID",
    ]
    dynamic_order: List[str] = []
    for record in tissue_entries:
        for key in record.keys():
            if key in base_order or key.startswith("_"):
                continue
            if key not in dynamic_order:
                dynamic_order.append(key)
    headers = base_order + dynamic_order

    items: List[Dict[str, Any]] = []
    for record in tissue_entries:
        record_copy = {key: value for key, value in record.items() if not key.startswith("_")}
        columns = {header: _value_to_string(record_copy.get(header)) or "" for header in headers}
        items.append({"columns": columns})
    return items, headers


def _build_cultivation_dataset(
    cultivation_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], List[Tuple[str, List[Dict[str, Any]]]]]:
    base_order = [
        "Plant ID",
        "Plant latin name",
        "Cultivation entry ID",
        "Cultivation entry name",
        "Cultivation detail URL",
        "Detail page ID",
    ]
    dynamic_order: List[str] = []
    for record in cultivation_entries:
        for key in record.keys():
            if key in base_order or key.startswith("_"):
                continue
            if key not in dynamic_order:
                dynamic_order.append(key)
    headers = base_order + dynamic_order

    items: List[Dict[str, Any]] = []
    downloads: List[Tuple[str, List[Dict[str, Any]]]] = []
    for record in cultivation_entries:
        record_copy = dict(record)
        photos = record_copy.pop("_photos", [])
        columns = {header: _value_to_string(record_copy.get(header)) or "" for header in headers}
        item: Dict[str, Any] = {"columns": columns}
        if photos:
            item["photos"] = photos
            identifier_parts = [
                columns.get("Plant ID"),
                columns.get("Cultivation entry ID"),
                columns.get("Cultivation entry name"),
            ]
            identifier = "_".join(part for part in identifier_parts if part)
            if not identifier:
                identifier = columns.get("Plant latin name") or "cultivation_entry"
            downloads.append((identifier, photos))
        items.append(item)
    return items, headers, downloads


def crawl_plant(
    crawler: "MPDBCrawler",
    include_details: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Crawl plant data and expose columns in the MPDBCrawler format."""
    delay_seconds = max(0.0, getattr(crawler, "delay", 0.5))
    raw_entries, tissue_entries, cultivation_entries = crawl_plants(
        limit=limit,
        delay_seconds=delay_seconds,
        session=crawler.session,
    )

    headers = EXPECTED_COLUMNS
    transformed: List[Dict[str, Any]] = []
    for entry in raw_entries:
        mapped_columns: Dict[str, str] = {}
        for column in headers:
            value = _value_to_string(entry.get(column))
            mapped_columns[column] = value if value is not None else ""

        identifier = mapped_columns.get("ID")
        photo_identifier = None
        if hasattr(crawler, "_derive_photo_identifier"):
            try:
                photo_identifier = crawler._derive_photo_identifier("plant", entry, None)
            except Exception:  # pragma: no cover - defensive against future refactors
                photo_identifier = None
        if not photo_identifier:
            photo_identifier = identifier or entry.get("Plant latin name") or entry.get("Common name")

        payload: Dict[str, Any] = {"columns": mapped_columns}
        if identifier:
            payload["id"] = identifier

        transformed.append(payload)

        allow_photo_download = True
        checker = getattr(crawler, "_should_download_plant_photos", None)
        if callable(checker):
            try:
                allow_photo_download = checker(photo_identifier)
            except Exception:  # pragma: no cover - defensive
                allow_photo_download = True

        if (
            getattr(crawler, "download_photos", False)
            and allow_photo_download
            and hasattr(crawler, "download_inaturalist_photos")
        ):
            plant_names: List[str] = []
            for key in ("Plant latin name", "Common name", "Crude drug latin name"):
                name_value = entry.get(key)
                if name_value:
                    plant_names.append(name_value)
            if plant_names and photo_identifier:
                crawler.download_inaturalist_photos(photo_identifier, plant_names)

    if tissue_entries:
        tissue_items, tissue_headers = _build_tissue_dataset(tissue_entries)
        if tissue_items:
            crawler.register_additional_result(
                category="tissue_culture_literature",
                items=tissue_items,
                headers=tissue_headers,
            )

    if cultivation_entries:
        cultivation_items, cultivation_headers, cultivation_downloads = _build_cultivation_dataset(cultivation_entries)
        if cultivation_items:
            if getattr(crawler, "download_photos", False) and cultivation_downloads:
                for identifier, photos in cultivation_downloads:
                    crawler._download_photos_for_category("cultivation", identifier, photos)
            crawler.register_additional_result(
                category="cultivation",
                items=cultivation_items,
                headers=cultivation_headers,
            )

    return transformed, headers


def crawl_tissue_culture_literature(
    crawler: "MPDBCrawler",
    include_details: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Crawl tissue culture literature via plant entries."""
    delay_seconds = max(0.0, getattr(crawler, "delay", 0.5))
    _, tissue_entries, _ = crawl_plants(limit=limit, delay_seconds=delay_seconds, session=crawler.session)
    items, headers = _build_tissue_dataset(tissue_entries)
    return items, headers


def crawl_cultivation(
    crawler: "MPDBCrawler",
    include_details: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Crawl cultivation entries via plant entries."""
    delay_seconds = max(0.0, getattr(crawler, "delay", 0.5))
    _, _, cultivation_entries = crawl_plants(limit=limit, delay_seconds=delay_seconds, session=crawler.session)
    items, headers, downloads = _build_cultivation_dataset(cultivation_entries)
    if getattr(crawler, "download_photos", False) and downloads:
        for identifier, photos in downloads:
            crawler._download_photos_for_category("cultivation", identifier, photos)
    return items, headers
