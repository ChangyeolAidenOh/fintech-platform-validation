"""
Stage 1 — Run all collectors sequentially.
Usage: python run_collectors.py
"""

# stdlib
import time

# local
from collectors.collector_sales import main as collect_sales
from collectors.collector_stores import main as collect_stores
from collectors.collector_traffic import main as collect_traffic
from collectors.collector_change import main as collect_change
from collectors.collector_facilities import main as collect_facilities
from collectors.collector_residents import main as collect_residents
from collectors.collector_ecos import main as collect_ecos

COLLECTORS = [
    ("Sales (Card-Transaction Based)", collect_sales),
    ("Stores (Target Variable)", collect_stores),
    ("Foot Traffic (External #1)", collect_traffic),
    ("District Change (External #2)", collect_change),
    ("Facilities (External #3)", collect_facilities),
    ("Residents", collect_residents),
    ("ECOS Macro (External #4)", collect_ecos),
]


def main():
    print("=" * 60)
    print("Stage 1: Running All Collectors")
    print("=" * 60)

    results = {}
    for label, func in COLLECTORS:
        print(f"\n{'=' * 60}")
        try:
            func()
            results[label] = "OK"
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            results[label] = f"FAILED: {e}"
        time.sleep(2)

    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    for label, status in results.items():
        print(f"  {label}: {status}")


if __name__ == "__main__":
    main()
