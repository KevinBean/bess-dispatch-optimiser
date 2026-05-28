"""Thorough UI walk of the live Cloud Run demo — exercises every control and
reports what each does, so bugs (e.g. regions not updating) are visible.

Run with the emptyos env's python (has playwright + chromium):
    python scripts/ui_walk.py
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://bess-dispatch-optimiser-293079882499.australia-southeast1.run.app/"
OUT = Path("D:/bess-dispatch-optimiser/docs")
OUT.mkdir(exist_ok=True)
REGIONS = ["SA1", "NSW1", "QLD1", "TAS1", "VIC1"]


def read_state(page) -> str:
    """Return the revenue metric, or any warning/error text shown in main area."""
    body = page.inner_text("body")
    # Streamlit metric value
    val = ""
    try:
        val = page.locator('[data-testid="stMetricValue"]').first.inner_text(timeout=2000)
    except Exception:
        pass
    flags = []
    for marker in ("No trained forecaster", "Traceback", "Error", "ModuleNotFound",
                   "FileNotFoundError", "KeyError"):
        if marker in body:
            flags.append(marker)
    return f"revenue={val!r} flags={flags}"


def select_region(page, region: str):
    """Open the Streamlit selectbox and pick a region."""
    sb = page.locator('[data-testid="stSelectbox"]').first
    sb.click()
    page.wait_for_timeout(400)
    page.get_by_role("option", name=region, exact=True).click()
    page.wait_for_timeout(6000)  # rerun + recompute


def main() -> int:
    report = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 1200})
        print("navigating (cold start may take ~30s)...")
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_selector("text=BESS Dispatch Optimiser", timeout=90_000)
        try:
            page.wait_for_selector("text=arbitrage revenue", timeout=90_000)
        except Exception:
            pass
        page.wait_for_timeout(2000)
        report.append(("load (SA1 default)", read_state(page)))

        for region in REGIONS:
            try:
                select_region(page, region)
                state = read_state(page)
            except Exception as e:  # noqa: BLE001
                state = f"INTERACTION FAILED: {e}"
            report.append((f"region={region}", state))
            page.screenshot(path=str(OUT / f"walk_region_{region}.png"))

        browser.close()

    print("\n===== UI WALK REPORT =====")
    for step, state in report:
        print(f"  {step:<22} {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
