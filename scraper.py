# from typing import List

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException


# class ReviewScraper:
#     """Selenium-based scraper for Amazon-like product review pages."""

#     def __init__(self, timeout: int = 15):
#         self.timeout = timeout

#     def scrape_reviews(self, url: str, max_reviews: int = 50) -> List[str]:
#         print(f"[DEBUG][scraper] Starting scrape for URL: {url}")
#         options = webdriver.ChromeOptions()
#         options.add_argument("--headless=new")
#         options.add_argument("--no-sandbox")
#         options.add_argument("--disable-dev-shm-usage")

#         driver = webdriver.Chrome(options=options)
#         reviews: List[str] = []

#         try:
#             driver.get(url)
#             wait = WebDriverWait(driver, self.timeout)

#             # Generic selectors for Amazon-like review blocks.
#             selectors = [
#                 "[data-hook='review-body'] span",
#                 ".review-text-content span",
#                 ".review-text",
#                 ".review-content",
#             ]

#             found_elements = []
#             for sel in selectors:
#                 try:
#                     wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, sel)))
#                     elems = driver.find_elements(By.CSS_SELECTOR, sel)
#                     if elems:
#                         found_elements = elems
#                         print(f"[DEBUG][scraper] Matched selector: {sel} count={len(elems)}")
#                         break
#                 except TimeoutException:
#                     print(f"[DEBUG][scraper] Selector timeout: {sel}")

#             for elem in found_elements[:max_reviews]:
#                 text = elem.text.strip()
#                 if text:
#                     reviews.append(text)

#             print(f"[DEBUG][scraper] Scraped {len(reviews)} reviews")
#             return reviews

#         finally:
#             driver.quit()





from typing import List, Set

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class ReviewScraper:
    """Selenium-based scraper for Amazon-like product review pages."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def scrape_reviews(self, url: str, max_reviews: int = 50) -> List[str]:
        print(f"[DEBUG][scraper] Starting scrape for URL: {url}")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        reviews: List[str] = []
        seen: Set[str] = set()

        try:
            driver.get(url)
            wait = WebDriverWait(driver, self.timeout)

            # Generic selectors for Amazon-like review blocks.
            selectors = [
                "[data-hook='review-body'] span",
                ".review-text-content span",
                ".review-text",
                ".review-content",
            ]

            active_selector = self._resolve_selector(driver, wait, selectors)
            if not active_selector:
                print("[DEBUG][scraper] No review selector matched.")
                return reviews

            print(f"[DEBUG][scraper] Using selector: {active_selector}")

            page_index = 1
            while len(reviews) < max_reviews:
                print(f"[DEBUG][scraper] Collecting page {page_index}")
                self._collect_page_reviews(driver, active_selector, reviews, seen, max_reviews)
                if len(reviews) >= max_reviews:
                    break

                moved = self._go_to_next_page(driver, wait)
                if not moved:
                    print("[DEBUG][scraper] No next page found. Stopping pagination.")
                    break
                page_index += 1

            print(f"[DEBUG][scraper] Scraped {len(reviews)} reviews (max requested={max_reviews})")
            return reviews

        finally:
            driver.quit()

    def _resolve_selector(self, driver, wait: WebDriverWait, selectors: List[str]) -> str | None:
        for sel in selectors:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, sel)))
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                if elems:
                    print(f"[DEBUG][scraper] Matched selector: {sel} count={len(elems)}")
                    return sel
            except TimeoutException:
                print(f"[DEBUG][scraper] Selector timeout: {sel}")
        return None

    def _collect_page_reviews(
        self,
        driver,
        selector: str,
        reviews: List[str],
        seen: Set[str],
        max_reviews: int,
    ) -> None:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        print(f"[DEBUG][scraper] Found {len(elements)} review elements on this page")
        for elem in elements:
            if len(reviews) >= max_reviews:
                return
            text = elem.text.strip()
            if text and text not in seen:
                seen.add(text)
                reviews.append(text)

    def _go_to_next_page(self, driver, wait: WebDriverWait) -> bool:
        next_selectors = [
            "li.a-last a",  # Amazon
            "a[aria-label='Next page']",
            "a.next",
            ".pagination-next a",
        ]
        for sel in next_selectors:
            try:
                next_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
                current_url = driver.current_url
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
                next_btn.click()
                wait.until(lambda d: d.current_url != current_url)
                print(f"[DEBUG][scraper] Navigated to next page using selector: {sel}")
                return True
            except (TimeoutException, NoSuchElementException):
                continue
            except Exception as exc:
                print(f"[DEBUG][scraper] Next-page click failed for {sel}: {exc}")
                continue
        return False