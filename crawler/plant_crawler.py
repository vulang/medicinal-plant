from __future__ import annotations

import time
import urllib.parse
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning
from urllib3.exceptions import NotOpenSSLWarning

if TYPE_CHECKING:  # pragma: no cover
    from mpdb_crawler import MPDBCrawler

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

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

    for tr in detail_table.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        label_text = _text_or_empty(tds[0]).lower()
        if label_text in mapping and len(tds) >= 2:
            key = mapping[label_text]
            value_cell = tds[1]
            links = value_cell.find_all("a")
            if links:
                value = ", ".join(_text_or_empty(a) for a in links)
            else:
                value = _text_or_empty(value_cell)
            data[key] = value

    growing = _parse_growing_data(detail_table)
    data.update(growing)

    return data


def iter_plant_entry_urls(session: Optional[requests.Session] = None) -> Iterable[str]:
    sess = session or _create_session()
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


def crawl_plants(limit: Optional[int] = None, delay_seconds: float = 0.5) -> List[Dict[str, str]]:
    session = _create_session()
    results: List[Dict[str, str]] = []

    count = 0
    for url in iter_plant_entry_urls(session=session):
        try:
            data = fetch_plant_detail(url, session=session)
            if data:
                results.append(data)
        except Exception:
            pass

        count += 1
        if limit is not None and count >= limit:
            break
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return results


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


def crawl_plant(
    crawler: "MPDBCrawler",
    include_details: bool = False,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Crawl plant data and expose columns in the MPDBCrawler format."""
    delay_seconds = max(0.0, getattr(crawler, "delay", 0.5))
    raw_entries = crawl_plants(limit=limit, delay_seconds=delay_seconds)

    headers = EXPECTED_COLUMNS
    transformed: List[Dict[str, Any]] = []
    for entry in raw_entries:
        mapped_columns: Dict[str, str] = {}
        for column in headers:
            value = _value_to_string(entry.get(column))
            mapped_columns[column] = value if value is not None else ""

        identifier = mapped_columns.get("ID")
        payload: Dict[str, Any] = {"columns": mapped_columns}
        if identifier:
            payload["id"] = identifier

        transformed.append(payload)

    return transformed, headers
