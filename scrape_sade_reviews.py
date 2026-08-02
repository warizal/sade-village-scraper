"""
GOOGLE MAPS REVIEWS SCRAPER FOR INDOBERT SENTIMENT ANALYSIS (GITHUB ACTIONS OPTIMIZED)
Target Location : Sade Village (Desa Sade), Lombok Tengah, NTB, Indonesia
Author          : Senior Python Data Engineer & ML Researcher
Language        : Python 3.13
"""

import os
import re
import time
import json
from datetime import datetime
from typing import List, Dict, Set, Optional, Any

import pandas as pd
import numpy as np
import openpyxl
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)


class GoogleMapsScraper:
    """Scraper profesional Google Maps Reviews untuk Dataset Penelitian Sentimen IndoBERT."""

    def __init__(
        self,
        target_url: str,
        output_csv: str = "sade_village_reviews_clean_5000.csv",
        checkpoint_file: str = "sade_checkpoint.json",
        target_count: int = 5000,
        autosave_interval: int = 100,
        max_no_change: int = 15
    ) -> None:
        self.target_url: str = target_url
        self.output_csv: str = output_csv
        self.checkpoint_file: str = checkpoint_file
        self.target_count: int = target_count
        self.autosave_interval: int = autosave_interval
        self.max_no_change: int = max_no_change

        self.driver: Optional[webdriver.Chrome] = None
        self.scraped_data: List[Dict[str, Any]] = []
        self.unique_hashes: Set[str] = set()

        logger.add(
            "scraping_process.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

    def _init_driver(self) -> None:
        """Inisialisasi Chrome WebDriver dengan opsi optimasi Virtual Display Xvfb / Cloud Server."""
        logger.info("Menginisialisasi Chrome WebDriver di Cloud Environment...")
        options = webdriver.ChromeOptions()

        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--lang=id-ID,id,en-US,en")

        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "translate": {"enabled": False}
        }
        options.add_experimental_option("prefs", prefs)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        logger.info("WebDriver berhasil diinisialisasi.")

    def _load_checkpoint(self) -> None:
        """Memuat checkpoint terakhir dari CSV dan JSON jika ada."""
        if os.path.exists(self.checkpoint_file) and os.path.exists(self.output_csv):
            try:
                df_existing = pd.read_csv(self.output_csv)
                self.scraped_data = df_existing.to_dict("records")

                for item in self.scraped_data:
                    hash_key = self._generate_hash(
                        str(item.get("nama_pengguna")),
                        str(item.get("ulasan_pengunjung")),
                        str(item.get("tanggal_review"))
                    )
                    self.unique_hashes.add(hash_key)

                logger.info(f"[RESUME] Berhasil memuat checkpoint! Total {len(self.scraped_data)} data.")
            except Exception as e:
                logger.error(f"Gagal memuat checkpoint: {e}. Memulai baru.")
                self.scraped_data = []
                self.unique_hashes = set()

    def _save_checkpoint(self) -> None:
        """Menyimpan dataset terkumpul ke file CSV."""
        if not self.scraped_data:
            # Buat file kosong berstruktur jika belum ada data sama sekali
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

    def _setup_page(self) -> None:
        """Membuka halaman target Google Maps."""
        if not self.driver:
            raise RuntimeError("Driver belum diaktifkan.")

        logger.info(f"Navigasi ke URL target Google Maps: {self.target_url}")
        self.driver.get(self.target_url)
        time.sleep(8)

        # Klik Tab Ulasan
        try:
            buttons = self.driver.find_elements(By.XPATH, "//button[contains(@aria-label, 'Ulasan') or contains(@aria-label, 'Reviews') or contains(., 'Ulasan')]")
            if buttons:
                self.driver.execute_script("arguments[0].click();", buttons[0])
                time.sleep(4)
        except Exception as e:
            logger.warning(f"Info tab ulasan: {e}")

    def _find_scrollable_container(0) -> Any:
        """Mencari kontainer scroll ulasan dengan multi-fallback."""
        scroll_script = """
            let containers = Array.from(document.querySelectorAll('div'));
            return containers.find(c => {
                let style = window.getComputedStyle(c);
                return (style.overflowY === 'auto' || style.overflowY === 'scroll') && c.scrollHeight > 400;
            });
        """
        try:
            container = self.driver.execute_script(scroll_script)
            if container:
                return container
        except Exception:
            pass

        selectors = [
            "//div[contains(@class, 'm6QE1c') and contains(@class, 'DshB1')]",
            "//div[@role='main']",
            "//div[contains(@class, 'm6QE1c')]"
        ]
        for xpath in selectors:
            elements = self.driver.find_elements(By.XPATH, xpath)
            for el in elements:
                try:
                    if el.size["height"] > 200:
                        return el
                except Exception:
                    continue

        return None

    def _expand_review_texts(self) -> None:
        try:
            more_buttons = self.driver.find_elements(
                By.XPATH,
                "//button[contains(@aria-label, 'Lainnya') or contains(text(), 'Lainnya') or contains(text(), 'More') or contains(text(), 'Selengkapnya')]"
            )
            for btn in more_buttons[:10]:
                if btn.is_displayed():
                    self.driver.execute_script("arguments[0].click();", btn)
        except Exception:
            pass

    @staticmethod
    def _clean_original_text(raw_text: str) -> str:
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

    def _extract_reviews_from_dom(self) -> int:
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        
        review_cards = soup.find_all("div", attrs={"data-review-id": True})
        if not review_cards:
            review_cards = soup.find_all("div", class_=lambda c: c and "jftiB" in c)

        new_records_added = 0
        for card in review_cards:
            try:
                name_el = card.find("div", class_=lambda c: c and ("d4rBO" in c or "header" in c or "d4r55" in c))
                nama_pengguna = name_el.text.strip() if name_el else "Anonym"

                rating = 0
                star_el = card.find("span", attrs={"aria-label": lambda a: a and ("bintang" in a.lower() or "star" in a.lower())})
                if star_el:
                    match = re.search(r"(\d+)", star_el["aria-label"])
                    if match:
                        rating = int(match.group(1))

                date_el = card.find("span", class_=lambda c: c and ("r0355b" in c or "rsqaWe" in c))
                if not date_el:
                    date_el = card.find("span", text=re.compile(r"(lalu|ago|bulan|tahun|minggu|hari)", re.I))
                tanggal_review = date_el.text.strip() if date_el else ""

                text_el = card.find("span", class_=lambda c: c and ("wi832d" in c or "MyT2O" in c or "wiI7pd" in c))
                if not text_el:
                    text_el = card.find("div", class_=lambda c: c and ("TOw1ed" in c or "MyEned" in c))
                
                raw_text = text_el.text.strip() if text_el else ""
                clean_original_text = self._clean_original_text(raw_text)

                if len(clean_original_text) < 3:
                    continue

                like_btn = card.find("button", attrs={"aria-label": lambda a: a and ("suka" in a.lower() or "like" in a.lower())})
                jumlah_suka = 0
                if like_btn and "aria-label" in like_btn.attrs:
                    match = re.search(r"(\d+)", like_btn["aria-label"].replace(".", "").replace(",", ""))
                    if match:
                        jumlah_suka = int(match.group(1))

                local_guide = "Tidak"
                lg_el = card.find(text=re.compile(r"Local Guide", re.I))
                if lg_el or card.find("span", class_=lambda c: c and "RfnDt" in c):
                    local_guide = "Ya"

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
                    new_records_added += 1

            except Exception as e:
                logger.debug(f"Parsing card error: {e}")
                continue

        return new_records_added

    def _scroll_action(self, scroll_container: Any) -> None:
        if scroll_container:
            try:
                self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight;", scroll_container)
                return
            except Exception:
                pass

        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def scrape(self) -> None:
        self._load_checkpoint()
        self._init_driver()

        try:
            self._setup_page()

            scroll_container = self._find_scrollable_container()
            no_new_data_counter = 0
            last_total_count = len(self.scraped_data)

            pbar = tqdm(total=self.target_count, initial=last_total_count, desc="Scraping Reviews Sade")

            while True:
                self._expand_review_texts()

                new_items_count = self._extract_reviews_from_dom()
                current_total_count = len(self.scraped_data)

                pbar.update(current_total_count - pbar.n)

                # Paksa simpan setiap menemukan data baru agar file CSV selalu ada
                if new_items_count > 0:
                    self._save_checkpoint()

                if current_total_count == last_total_count:
                    no_new_data_counter += 1
                    logger.warning(f"Scroll dilakukan, belum ada ulasan baru ({no_new_data_counter}/{self.max_no_change})")

                    if no_new_data_counter % 3 == 0:
                        scroll_container = self._find_scrollable_container()
                else:
                    no_new_data_counter = 0
                    last_total_count = current_total_count

                if no_new_data_counter >= self.max_no_change:
                    logger.info("Ulasan telah habis atau mencapai batas percobaan scroll.")
                    break

                if current_total_count >= self.target_count:
                    logger.info(f"Target minimal {self.target_count} dataset bersih tercapai!")
                    break

                self._scroll_action(scroll_container)
                time.sleep(3.0)

            pbar.close()
            self._save_checkpoint()
            logger.success(f"PROSES SELESAI! Total dataset original Desa Sade: {len(self.scraped_data)} data.")

        except Exception as e:
            logger.critical(f"Terjadi error fatal selama proses scraping: {e}")
            self._save_checkpoint()
        finally:
            if self.driver:
                self.driver.quit()


if __name__ == "__main__":
    URL_DESA_SADE = (
        "https://www.google.com/maps/place/Sade+Village/"
        "@-8.839218,116.2917036,17z/data=!4m8!3m7!"
        "1s0x2dcdb2586da21cb7:0xa9bc1ee2bd2437bf!"
        "8m2!3d-8.839218!4d116.2942785!9m1!1b1!16s%2Fg%2F11b6vh993q?entry=ttu"
    )

    scraper = GoogleMapsScraper(
        target_url=URL_DESA_SADE,
        output_csv="sade_village_reviews_clean_5000.csv",
        checkpoint_file="sade_checkpoint.json",
        target_count=5000,
        autosave_interval=50,
        max_no_change=15
    )

    scraper.scrape()
