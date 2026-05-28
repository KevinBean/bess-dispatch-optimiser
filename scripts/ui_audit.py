"""Critical UI audit pass — capture the fold + check the backtest tab is region-aware."""
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://bess-dispatch-optimiser-293079882499.australia-southeast1.run.app/"
OUT = Path("D:/bess-dispatch-optimiser/docs")


def main():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        # desktop fold (not full_page) to see whitespace/layout as a visitor does
        pg = b.new_page(viewport={"width": 1366, "height": 768})
        pg.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        pg.wait_for_selector("text=arbitrage revenue", timeout=90_000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "audit_fold.png"))
        # check the truncated metric text
        metrics = pg.locator('[data-testid="stMetricValue"]').all_inner_texts()
        print("metric values:", metrics)

        # switch to QLD1, open Backtest — is the chart region-aware?
        pg.locator('[data-testid="stSelectbox"]').first.click(); pg.wait_for_timeout(400)
        pg.get_by_role("option", name="QLD1", exact=True).click(); pg.wait_for_timeout(6000)
        pg.get_by_role("tab", name="Backtest").click(); pg.wait_for_timeout(2500)
        body = pg.inner_text("body")
        print("backtest mentions SA1 while QLD1 selected:", "SA1" in body)
        pg.screenshot(path=str(OUT / "audit_backtest_qld1.png"), full_page=True)

        # mobile viewport check
        m = b.new_page(viewport={"width": 390, "height": 844})
        m.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        m.wait_for_timeout(8000)
        m.screenshot(path=str(OUT / "audit_mobile.png"))
        b.close()


if __name__ == "__main__":
    main()
