"""
Seoul Open Data API utility.
Shared HTTP client for all collectors fetching from data.seoul.go.kr.
"""

# stdlib
import os
import time

# third-party
import requests
from dotenv import load_dotenv

load_dotenv()

SEOUL_API_KEY = os.getenv("SEOUL_API_KEY")
SEOUL_BASE_URL = "http://openapi.seoul.go.kr:8088"
CALL_INTERVAL = 1.0
PAGE_SIZE = 1000


def fetch_seoul_api(service_name, start=1, end=PAGE_SIZE):
    """Fetch a single page from Seoul Open Data API. Returns list of rows or None."""
    if not SEOUL_API_KEY:
        print("[ERROR] SEOUL_API_KEY not set in .env")
        return None

    url = f"{SEOUL_BASE_URL}/{SEOUL_API_KEY}/json/{service_name}/{start}/{end}/"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"  [ERROR] HTTP {e.response.status_code}: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None

    # Seoul API wraps data in a key matching the service name
    if service_name in data:
        result_block = data[service_name]
        # Check for API-level errors
        code = result_block.get("RESULT", {}).get("CODE", "")
        if code == "INFO-200":
            print(f"  [WARN] No data for {service_name} [{start}:{end}]")
            return []
        if code != "INFO-000":
            msg = result_block.get("RESULT", {}).get("MESSAGE", "Unknown error")
            print(f"  [ERROR] API error {code}: {msg}")
            return None
        total = result_block.get("list_total_count", 0)
        rows = result_block.get("row", [])
        return rows
    else:
        # Check if it's an error response
        if "RESULT" in data:
            code = data["RESULT"].get("CODE", "")
            msg = data["RESULT"].get("MESSAGE", "")
            print(f"  [ERROR] API error {code}: {msg}")
        else:
            print(f"  [ERROR] Unexpected response structure: {list(data.keys())}")
        return None


def fetch_all_pages(service_name, page_size=PAGE_SIZE):
    """Fetch all pages from a Seoul API endpoint. Returns combined list of rows."""
    print(f"Fetching {service_name}...")

    # First page to get total count
    first_url = f"{SEOUL_BASE_URL}/{SEOUL_API_KEY}/json/{service_name}/1/{page_size}/"
    try:
        resp = requests.get(first_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] First page failed: {e}")
        return []

    if service_name not in data:
        if "RESULT" in data:
            msg = data["RESULT"].get("MESSAGE", "")
            print(f"  [ERROR] {msg}")
        return []

    result_block = data[service_name]
    total = result_block.get("list_total_count", 0)
    rows = result_block.get("row", [])

    if total == 0:
        print(f"  No data found")
        return []

    print(f"  Total rows: {total}")
    all_rows = list(rows)

    # Remaining pages
    fetched = len(rows)
    while fetched < total:
        start = fetched + 1
        end = min(fetched + page_size, total)
        time.sleep(CALL_INTERVAL)

        page_rows = fetch_seoul_api(service_name, start, end)
        if page_rows is None:
            print(f"  [WARN] Stopping at {fetched}/{total} due to error")
            break
        all_rows.extend(page_rows)
        fetched += len(page_rows)
        if fetched % 25000 < page_size or fetched >= total:
            print(f"  Progress: {fetched}/{total}")

    print(f"  Fetched {len(all_rows)} rows")
    return all_rows


def get_total_count(service_name):
    """Get total row count for a service without fetching all data."""
    rows = fetch_seoul_api(service_name, 1, 1)
    if rows is None:
        return -1

    url = f"{SEOUL_BASE_URL}/{SEOUL_API_KEY}/json/{service_name}/1/1/"
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if service_name in data:
            return data[service_name].get("list_total_count", 0)
    except Exception:
        pass
    return -1
