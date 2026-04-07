import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST_FOLDER = r"DOP_data"
BASE_URL = "https://opengeodata.lgl-bw.de/data/dop20"
MAX_WORKERS = 10
RETRIES = 3
TIMEOUT = 120

# Replace with fresh cookie if needed
COOKIE = (
    "MyCookie=YOUR_COOKIE_HERE; "
    "TS0126c163=YOUR_TS_COOKIE_HERE"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Cookie": COOKIE,
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Priority": "u=0, i",
}

os.makedirs(DEST_FOLDER, exist_ok=True)

# Erstellung von Tiles-Liste. Diese Zahlen ggf. für andere Region anpassen.
tiles = []
for i in range(421, 492, 2):
    for j in range(5396, 5453, 2):
        filename = f"dop20rgb_32_{i}_{j}_2_bw.zip"
        url = f"{BASE_URL}/{filename}"
        filepath = os.path.join(DEST_FOLDER, filename)
        tiles.append((url, filepath))

print(f"Total tiles to process: {len(tiles)}")

def download_tile(url, filepath):
    if os.path.exists(filepath):
        print(f"Skipping existing: {os.path.basename(filepath)}")
        return

    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded: {os.path.basename(filepath)}")
            return
        except Exception as e:
            print(f"Attempt {attempt} failed for {os.path.basename(filepath)}: {e}")
            if attempt < RETRIES:
                time.sleep(5)
            else:
                print(f"FAILED permanently: {os.path.basename(filepath)}")


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_tile, url, path) for url, path in tiles]
    for _ in as_completed(futures):
        pass

print("All downloads processed.")