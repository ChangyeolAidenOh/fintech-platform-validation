"""
Stage 1 — API endpoint verification spike.
Run this FIRST to confirm Seoul Open Data service names work.
Usage: python spike_api_check.py
"""

# stdlib
import os
import sys

# third-party
import requests
from dotenv import load_dotenv

load_dotenv()

SEOUL_API_KEY = os.getenv("SEOUL_API_KEY")
SEOUL_BASE_URL = "http://openapi.seoul.go.kr:8088"

# ================================================================
# Service name candidates for each dataset
# If the primary name fails, try alternates
# ================================================================
SERVICE_CANDIDATES = {
    "추정매출(상권)": [
        "VwsmTrdarSelngQq",
        "tbgisTrdarRelm",
    ],
    "점포(상권)": [
        "VwsmTrdarStorQq",
        "tbgisTrdarStorQq",
    ],
    "유동인구(상권배후지)": [
        "VwsmTrdarFlpopQq",
        "VwsmTrdarWrcFlpopQq",
    ],
    "상권변화지표(상권)": [
        "VwsmTrdarChgIndQq",
        "tbgisTrdarChgInd",
    ],
    "집객시설(상권)": [
        "VwsmTrdarFcltyQq",
        "tbgisTrdarFclty",
    ],
    "상주인구(상권배후지)": [
        "VwsmTrdarPopltnQq",
        "VwsmTrdarWrcPopltnQq",
    ],
}


def check_service(label, candidates):
    """Try each candidate service name and report which one works."""
    for name in candidates:
        url = f"{SEOUL_BASE_URL}/{SEOUL_API_KEY}/json/{name}/1/5/"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()

            if name in data:
                total = data[name].get("list_total_count", 0)
                rows = data[name].get("row", [])
                code = data[name].get("RESULT", {}).get("CODE", "")

                if code == "INFO-000" and total > 0:
                    print(f"  [OK] {label}: {name} -> {total} rows")
                    if rows:
                        print(f"       Sample keys: {list(rows[0].keys())[:8]}...")
                    return name
                else:
                    print(f"  [--] {label}: {name} -> code={code}, total={total}")
            elif "RESULT" in data:
                code = data["RESULT"].get("CODE", "")
                msg = data["RESULT"].get("MESSAGE", "")
                print(f"  [--] {label}: {name} -> {code}: {msg}")
            else:
                print(f"  [--] {label}: {name} -> unexpected response")
        except Exception as e:
            print(f"  [--] {label}: {name} -> {e}")

    print(f"  [FAIL] {label}: no working service name found")
    return None


def main():
    if not SEOUL_API_KEY:
        print("[ERROR] SEOUL_API_KEY not set in .env")
        print("1. Go to https://data.seoul.go.kr")
        print("2. Sign up and get an API key")
        print("3. Add SEOUL_API_KEY=your_key to .env")
        sys.exit(1)

    print("=" * 60)
    print("Seoul Open Data API Endpoint Verification")
    print("=" * 60)

    working = {}
    for label, candidates in SERVICE_CANDIDATES.items():
        result = check_service(label, candidates)
        if result:
            working[label] = result
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Working: {len(working)}/{len(SERVICE_CANDIDATES)}")
    for label, name in working.items():
        print(f"  {label}: {name}")

    failed = set(SERVICE_CANDIDATES.keys()) - set(working.keys())
    if failed:
        print(f"\nFailed: {failed}")
        print("Check https://data.seoul.go.kr for correct service names")
        print("Update SERVICE_NAME constants in each collector accordingly")

    # Write config file for collectors
    config_path = "collectors/seoul_services.py"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated Seoul API service name config."""\n\n')
        for label, name in working.items():
            safe_label = label.split("(")[0].strip()
            var_name = {
                "추정매출": "SALES_SERVICE",
                "점포": "STORES_SERVICE",
                "유동인구": "TRAFFIC_SERVICE",
                "상권변화지표": "CHANGE_SERVICE",
                "집객시설": "FACILITIES_SERVICE",
                "상주인구": "RESIDENTS_SERVICE",
            }.get(safe_label, f"UNKNOWN_{safe_label}")
            f.write(f'{var_name} = "{name}"\n')
    print(f"\nConfig written to {config_path}")


if __name__ == "__main__":
    main()
