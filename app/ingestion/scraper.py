"""
BeautifulSoup web scraper for extracting text and tables from URLs.
"""
import time
import logging
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AntinoRAGBot/1.0)"
}

SKIP_TAGS = {
    "script", "style", "noscript", "head", "meta",
    "link", "svg", "img", "button", "nav", "footer"
}


def scrape_url(url: str, timeout: int = 15) -> Dict:
    """
    Scrape a single URL. Returns dict with text, tables, source.
    """
    t0 = time.time()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"[Scraper] Failed to fetch {url}: {e}")
        return {"source": url, "text": "", "tables": [], "error": str(e)}

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove unwanted tags
    for tag in soup(SKIP_TAGS):
        tag.decompose()

    # Extract main text
    text = soup.get_text(separator=" ", strip=True)
    # Clean up whitespace
    import re
    text = re.sub(r'\s+', ' ', text).strip()

    # Extract tables as plain text rows
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables.append("\n".join(rows))

    elapsed = time.time() - t0
    logger.info(f"[Scraper] {url} → {len(text)} chars, {len(tables)} tables in {elapsed:.2f}s")
    return {"source": url, "text": text, "tables": tables}


def crawl_site(base_url: str, max_pages: int = 30) -> List[Dict]:
    """
    Crawl all internal links of a site up to max_pages.
    Returns list of scraped page dicts.
    """
    base_domain = urlparse(base_url).netloc
    visited = set()
    to_visit = [base_url]
    results = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        data = scrape_url(url)
        if data.get("text"):
            results.append(data)

        # Find internal links to crawl
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "lxml")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                # Only crawl same domain, http(s), no fragments
                if (
                    parsed.netloc == base_domain
                    and parsed.scheme in ("http", "https")
                    and full_url not in visited
                    and "#" not in full_url
                    and "?" not in full_url
                ):
                    to_visit.append(full_url)
        except Exception:
            pass

    logger.info(f"[Scraper] Crawled {len(results)} pages from {base_url}")
    return results
