import requests
from bs4 import BeautifulSoup
import json
import os

DOCS = {
    "mifid2": "32014L0065",
    "psd2": "32015L2366",
    "gdpr": "32016R0679",
    "dora": "32022R2554"
}

def scrape_eurlex():
    corpus = []
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    for source_file, celex in DOCS.items():
        print(f"Fetching {source_file} (CELEX: {celex})...")
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"  Failed to fetch {celex} (HTTP {resp.status_code})")
                continue
                
            soup = BeautifulSoup(resp.content, "html.parser")
            
            # EUR-Lex uses <p class="oj-ti-art"> or <p class="ti-art"> for "Chunk X"
            # and <p class="oj-sti-art"> or <p class="sti-art"> for the article title.
            article_nodes = soup.find_all(lambda tag: tag.name in ["p", "div", "span"] and tag.get("class") and any(cls in tag.get("class") for cls in ["ti-art", "oj-ti-art"]))
            
            print(f"  Found {len(article_nodes)} articles in {source_file}")
            
            for node in article_nodes:
                art_text_parts = []
                chunk_id = node.get_text(strip=True).replace("Chunk", "").strip()
                
                # The next sibling is usually the title
                curr = node.find_next_sibling()
                section_heading = ""
                if curr and curr.name in ["p", "div"] and curr.get("class") and any(cls in curr.get("class") for cls in ["sti-art", "oj-sti-art", "eli-title"]):
                    section_heading = curr.get_text(strip=True)
                    curr = curr.find_next_sibling()
                
                # Read all following siblings until the next 'ti-art' or 'oj-ti-art'
                while curr:
                    if curr.name in ["p", "div", "span"] and curr.get("class") and any(cls in curr.get("class") for cls in ["ti-art", "oj-ti-art", "ti-section-1", "ti-chapter", "oj-ti-section-1", "oj-ti-chapter"]):
                        break
                    # Only extract text from normal paragraphs or lists
                    if curr.name in ["p", "table", "ul", "ol", "div"]:
                        text = curr.get_text(" ", strip=True)
                        if text:
                            art_text_parts.append(text)
                    curr = curr.find_next_sibling()
                    
                full_text = " ".join(art_text_parts)
                if full_text and chunk_id:
                    corpus.append({
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "section_heading": section_heading,
                        "text": full_text,
                        "source_url": url
                    })
                    
        except Exception as e:
            print(f"  Error parsing {source_file}: {e}")

    # Save to corpus.jsonl
    with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSuccessfully scraped {len(corpus)} total articles across {len(DOCS)} regulations.")
    print("Saved to data/corpus.jsonl")

if __name__ == "__main__":
    scrape_eurlex()
