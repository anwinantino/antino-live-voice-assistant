"""
Ingest the full Antino website into Pinecone.
Run once after creating the index: python scripts/ingest_antino.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.ingestion.scraper import crawl_site
from app.ingestion.processor import process_scraped_pages
from app.ingestion.embedder import upsert_chunks

ANTINO_URL = "https://www.antino.com/"
MAX_PAGES = 40


def main():
    print(f"\n{'='*60}")
    print(f"  Antino RAG — Full Website Ingestion")
    print(f"{'='*60}")
    print(f"Target: {ANTINO_URL} (max {MAX_PAGES} pages)\n")

    total_start = time.time()

    # Step 1: Crawl
    print("Step 1/3: Crawling website...")
    t0 = time.time()
    pages = crawl_site(ANTINO_URL, max_pages=MAX_PAGES)
    print(f"  ✅ Scraped {len(pages)} pages in {time.time()-t0:.1f}s")

    if not pages:
        print("  ❌ No pages scraped. Check URL or network.")
        return

    # Step 2: Process
    print("\nStep 2/3: Processing and chunking text...")
    t0 = time.time()
    chunks = process_scraped_pages(pages)
    print(f"  ✅ Generated {len(chunks)} chunks in {time.time()-t0:.1f}s")

    if not chunks:
        print("  ❌ No chunks generated.")
        return

    # Step 3: Embed & Upsert
    print(f"\nStep 3/3: Embedding and indexing {len(chunks)} chunks to Pinecone...")
    t0 = time.time()

    from tqdm import tqdm
    batch_size = 100
    total_upserted = 0

    with tqdm(total=len(chunks), desc="Indexing", unit="chunks") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            upserted = upsert_chunks(batch)
            total_upserted += upserted
            pbar.update(len(batch))

    elapsed = time.time() - t0
    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  ✅ Ingestion Complete!")
    print(f"     Pages scraped:  {len(pages)}")
    print(f"     Chunks indexed: {total_upserted}")
    print(f"     Time taken:     {total_elapsed:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
