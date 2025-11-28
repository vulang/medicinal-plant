# MPDB Crawler

Command line crawler for the Comprehensive Medicinal Plant Database (MPDB) hosted at `mpdb.nibiohn.go.jp`. It can export tabular data from any supported category, follow detail pages to gather richer metadata, and optionally download accompanying photos.

## Features

- Fetch list data for categories such as `plant`, `crudedrug`, `jplist`, `syohou`, `compound`, `sample`, `gene`, `lcms`, `jp_identification`, and `jp_assay`.
- Output scraped content as JSON (default) or CSV, with CSV helpers that flatten nested link information.
- Follow detail pages (`--include-details`) to capture structured rows, links, and images.
- Download photo assets for plants, samples, or compounds into a predictable directory tree.
- Throttle and timeout controls to stay within polite crawling limits.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt` (`requests`, `beautifulsoup4`)

Install dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The crawler uses HTTP requests; ensure outbound network access is available when running it.

## Usage

Run the crawler with default settings (plants, JSON to stdout):

```bash
python mpdb_crawler.py
```

Write JSON for specific categories to a file:

```bash
python mpdb_crawler.py \
  --category plant \
  --category sample \
  --include-details \
  --output data/mpdb.json
```

Export CSV (one category per file when a directory is provided):

```bash
python mpdb_crawler.py \
  --category compound \
  --format csv \
  --output csv_out
```

Download photos for plants alongside detail scraping:

```bash
python mpdb_crawler.py \
  --category plant \
  --include-details \
  --download-photos \
  --photo-dir photos
```

Useful flags:

- `--limit N` pull only the first `N` rows (handy for testing).
- `--delay SECONDS` adjust the pause between requests (default `0.5`).
- `--timeout SECONDS` change the HTTP timeout (default `30`).
- `--low-support-report path/to/classification_report.txt` only download plant photos for class IDs whose support is below `--low-support-threshold` (default `50`), useful for backfilling underrepresented classes from a model report.
- `-v/--verbose` emit debug logging.

When using CSV output and multiple categories, pass a directory path to `--output`; per-category CSV files will be created inside it.

## Output Structure

- **JSON**: array of `{ "category": "...", "items": [...] }` objects. Each item captures the source row number, normalized columns, optional detail payload, and detail URL.
- **CSV**: headers consist of the row number plus the order surfaced by the MPDB table. Nested link data is flattened to text (or URLs when text is absent).

Photos (when enabled) are stored under `photos/<category>/<identifier>/` where the identifier is derived from the plant/sample/compound metadata.

## Notes

- The script respects polite crawling defaults but you remain responsible for complying with MPDB usage policies.
- Network failures or server-side protections can surface as `requests` exceptions; rerunning with `-v` can help diagnose issues.
- `jp_identification` and `jp_assay` lack list views; the crawler requests detail pages directly (IDs 1-676 for identification, 1-50 for assay).
