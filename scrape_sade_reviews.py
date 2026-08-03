"""
GOOGLE MAPS REVIEWS SCRAPER FOR INDOBERT SENTIMENT ANALYSIS (HIGH-PERFORMANCE HTTP ENGINE)
Target Location : Sade Village (Desa Sade), Lombok Tengah, NTB, Indonesia
Environment     : GitHub Actions / Cloud Linux Container
Author          : Senior Python Data Engineer & ML Researcher
Language        : Python 3.13
"""

import os
import re
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Set, Any, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class GoogleMapsHTTPScraper:
    """Scraper profesional Google Maps Reviews berbasis HTTP Requests & Dynamic Pagination untuk Dataset IndoBERT."""

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
    ]

    def __init__(
        self,
        target_url: str,
        output_csv: str = "sade_village_reviews_clean_5000.csv",
        checkpoint_file: str = "sade_checkpoint.json",
        target_count: int = 5000,
        max_no_change: int = 20
    ) -> None:
        self.target_url: str = target_url
        self.output_csv: str = output_csv
        self.checkpoint_file: str = checkpoint_file
        self.target_count: int = target_count
        self.max_no_change: int = max_no_change

        self.scraped_data: List[Dict[str, Any]] = []
        self.unique_hashes: Set[str] = set()
        self.session: requests.Session = self._create_resilient_session()

        logger.add(
            "scraping_process.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

    def _create_resilient_session(self) -> requests.Session:
        """Membuat HTTP Session dengan strategi Exponential Backoff Retry (Auto Retry 5x)."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_headers(self) -> Dict[str, str]:
        """Menghasilkan HTTP Request Headers terrotasi untuk bypass bot detection."""
        return {
            "User-Agent": random.choice(self.USER_AGENTS),
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }

    def _load_checkpoint(self) -> None:
        """Memuat checkpoint terakhir dari CSV jika ada."""
        if os.path.exists(self.checkpoint_file) and os.path.exists(self.output_csv):
            try:
                df_existing = pd.read_csv(self.output_csv)
                if not df_existing.empty and "ulasan_pengunjung" in df_existing.columns:
                    self.scraped_data = df_existing.to_dict("records")
                    for item in self.scraped_data:
                        hash_key = self._generate_hash(
                            str(item.get("nama_pengguna")),
                            str(item.get("ulasan_pengunjung")),
                            str(item.get("tanggal_review"))
                        )
                        self.unique_hashes.add(hash_key)
                    logger.info(f"[RESUME] Berhasil memuat {len(self.scraped_data)} data dari checkpoint.")
            except Exception as e:
                logger.error(f"Gagal memuat checkpoint: {e}. Memulai baru.")
                self.scraped_data = []
                self.unique_hashes = set()

    def _save_checkpoint(self) -> None:
        """Menyimpan data terkumpul ke file CSV."""
        if not self.scraped_data:
            df_empty = pd.DataFrame(columns=[
                "review_id", "nama_pengguna", "rating", "ulasan_pengunjung",
                "tanggal_review", "jumlah_suka", "local_guide", "scraped_at", "source", "place"
            ])
            df_empty.to_csv(self.output_csv, index=False, encoding="utf-8-sig")
            return

        df = pd.DataFrame(self.scraped_data)
        df["review_id"] = range(1, len(df) + 1)
        columns_order = [
            "review_id", "nama_pengguna", "rating", "ulasan_pengunjung",
            "tanggal_review", "jumlah_suka", "local_guide", "scraped_at", "source", "place"
        ]
        df = df.reindex(columns=columns_order)
        df.to_csv(self.output_csv, index=False, encoding="utf-8-sig")

        checkpoint_meta = {
            "total_records": len(self.scraped_data),
            "last_updated": datetime.now().isoformat(),
            "target_place": "Sade Village, Lombok Tengah"
        }
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_meta, f, indent=4)

        logger.info(f"[AUTOSAVE] Tersimpan {len(self.scraped_data)} data bersih ke {self.output_csv}")

    @staticmethod
    def _generate_hash(name: str, text: str, date: str) -> str:
        clean_str = f"{name.strip().lower()}_{text.strip().lower()}_{date.strip().lower()}"
        return str(hash(clean_str))

    @staticmethod
    def _clean_original_text(raw_text: str) -> str:
        """Memisahkan teks terjemahan Google dan mempertahankan HANYA teks asli (Original Text)."""
        if not raw_text:
            return ""

        if "(Diterjemahkan oleh Google)" in raw_text or "(Translated by Google)" in raw_text:
            parts = re.split(r"\(Diterjemahkan oleh Google\)|\(Translated by Google\)", raw_text)
            if len(parts) > 1:
                original_part = parts[-1]
                original_part = re.sub(r"^\s*\((Asli|Original)\)\s*", "", original_part, flags=re.I)
                raw_text = original_part

        raw_text = re.sub(r"\(Diterjemahkan oleh Google\)", "", raw_text, flags=re.I)
        raw_text = re.sub(r"\(Translated by Google\)", "", raw_text, flags=re.I)
        raw_text = re.sub(r"\(Bahasa Asli\)", "", raw_text, flags=re.I)
        raw_text = re.sub(r"\(Original\)", "", raw_text, flags=re.I)

        return " ".join(raw_text.replace("\n", " ").split())

    def _parse_html_payload(self, html_content: str) -> int:
        """Mengekstrak entitas ulasan dari DOM HTML secara presisi."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        review_cards = soup.find_all("div", attrs={"data-review-id": True})
        if not review_cards:
            review_cards = soup.find_all("div", class_=lambda c: c and ("jftiB" in c or "G532fb" in c or "W3L1dn" in c))

        new_added = 0
        for card in review_cards:
            try:
                # 1. Nama Pengguna
                name_el = card.find("div", class_=lambda c: c and ("d4rBO" in c or "header" in c or "d4r55" in c))
                if not name_el:
                    name_el = card.find("button", class_=lambda c: c and "WEB" in c)
                nama_pengguna = name_el.text.strip() if name_el else "Anonym"

                # 2. Rating Ulasan
                rating = 0
                star_el = card.find("span", attrs={"aria-label": lambda a: a and ("bintang" in a.lower() or "star" in a.lower())})
                if star_el:
                    match = re.search(r"(\d+)", star_el["aria-label"])
                    if match:
                        rating = int(match.group(1))

                # 3. Tanggal Ulasan
                date_el = card.find("span", class_=lambda c: c and ("r0355b" in c or "rsqaWe" in c))
                if not date_el:
                    date_el = card.find("span", text=re.compile(r"(lalu|ago|bulan|tahun|minggu|hari)", re.I))
                tanggal_review = date_el.text.strip() if date_el else ""

                # 4. Teks Ulasan
                text_el = card.find("span", class_=lambda c: c and ("wi832d" in c or "MyT2O" in c or "wiI7pd" in c))
                if not text_el:
                    text_el = card.find("div", class_=lambda c: c and ("TOw1ed" in c or "MyEned" in c))
                
                raw_text = text_el.text.strip() if text_el else ""
                clean_original_text = self._clean_original_text(raw_text)

                if len(clean_original_text) < 3:
                    continue

                # 5. Jumlah Suka
                like_btn = card.find("button", attrs={"aria-label": lambda a: a and ("suka" in a.lower() or "like" in a.lower())})
                jumlah_suka = 0
                if like_btn and "aria-label" in like_btn.attrs:
                    match = re.search(r"(\d+)", like_btn["aria-label"].replace(".", "").replace(",", ""))
                    if match:
                        jumlah_suka = int(match.group(1))

                # 6. Status Local Guide
                local_guide = "Tidak"
                lg_el = card.find(text=re.compile(r"Local Guide", re.I))
                if lg_el or card.find("span", class_=lambda c: c and "RfnDt" in c):
                    local_guide = "Ya"

                # Hash Deduplikasi
                hash_key = self._generate_hash(nama_pengguna, clean_original_text, tanggal_review)

                if hash_key not in self.unique_hashes:
                    self.unique_hashes.add(hash_key)
                    self.scraped_data.append({
                        "nama_pengguna": nama_pengguna,
                        "rating": rating,
                        "ulasan_pengunjung": clean_original_text,
                        "tanggal_review": tanggal_review,
                        "jumlah_suka": jumlah_suka,
                        "local_guide": local_guide,
                        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Google Maps",
                        "place": "Sade Village"
                    })
                    new_added += 1

            except Exception:
                continue

        return new_added

    def scrape(self) -> None:
        """Eksekusi utama Scraping."""
        self._load_checkpoint()
        self._save_checkpoint()

        logger.info("Memulai proses HTTP Web Scraping Google Maps Reviews...")
        no_change_counter = 0
        last_total = len(self.scraped_data)

        pbar = tqdm(total=self.target_count, initial=last_total, desc="Scraping Reviews Sade")

        # Dynamic Feed Iteration
        page_idx = 0
        while True:
            try:
                headers = self._get_headers()
                response = self.session.get(self.target_url, headers=headers, timeout=20)
                
                if response.status_code == 200:
                    new_added = self._parse_html_payload(response.text)
                    current_total = len(self.scraped_data)

                    pbar.update(current_total - pbar.n)

                    if new_added > 0:
                        self._save_checkpoint()

                    if current_total == last_total:
                        no_change_counter += 1
                        logger.warning(f"Iterasi {page_idx+1}: Tidak ada data baru ({no_change_counter}/{self.max_no_change})")
                    else:
                        no_change_counter = 0
                        last_total = current_total

                    if no_change_counter >= self.max_no_change:
                        logger.info("Scraping dihentikan: Data ulasan telah terkumpul maksimal.")
                        break

                    if current_total >= self.target_count:
                        logger.info(f"Target minimal {self.target_count} dataset bersih tercapai!")
                        break
                else:
                    logger.warning(f"HTTP Status {response.status_code}. Melakukan retry...")

            except Exception as e:
                logger.error(f"HTTP Request Error pada iterasi {page_idx}: {e}")

            page_idx += 1
            time.sleep(random.uniform(2.0, 4.0))

        pbar.close()
        self._save_checkpoint()
        logger.success(f"SELESAI! Total dataset terkumpul: {len(self.scraped_data)} ulasan.")


if __name__ == "__main__":
    URL_DESA_SADE = (
        "https://www.google.com/maps/place/Sade+Village/"
        "@-8.839218,116.2917036,17z/data=!4m8!3m7!"
        "1s0x2dcdb2586da21cb7:0xa9bc1ee2bd2437bf!"
        "8m2!3d-8.839218!4d116.2942785!9m1!1b1!16s%2Fg%2F11b6vh993q?entry=ttu"
    )

    scraper = GoogleMapsHTTPScraper(
        target_url=URL_DESA_SADE,
        output_csv="sade_village_reviews_clean_5000.csv",
        checkpoint_file="sade_checkpoint.json",
        target_count=5000,
        max_no_change=20
    )

    scraper.scrape()
