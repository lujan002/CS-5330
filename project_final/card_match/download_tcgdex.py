"""Download the full English Pokemon TCG card gallery from TCGdex.

The set endpoint already returns every card's id, name and image base URL, so a
full pull is ~220 metadata requests plus one image request per card instead of a
per-card detail request. National dex numbers only exist on the card detail
endpoint, so those are fetched separately behind --with-dex.

Resumable: images already on disk are skipped, so re-running after an
interruption only fetches what is missing.

    python3 -m card_match.download_tcgdex
    python3 -m card_match.download_tcgdex --with-dex --workers 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .common import IMAGE_DIR, META_PATH, TCG_DIR

API = "https://api.tcgdex.net/v2/en"

# Image assets are served as <base>/<quality>.<extension>. high.webp is 600x825
# at ~65 KB, about a fifth the size of high.png for the same pixels.
DEFAULT_QUALITY = "high"
DEFAULT_EXTENSION = "webp"


def make_session(retries: int = 4) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=64, pool_maxsize=64, max_retries=retries
    )
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "cs5330-card-match/1.0"
    return session


def get_json(session: requests.Session, url: str, attempts: int = 4):
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            # 429/5xx are worth another try; anything else is not.
            if response.status_code not in (429, 500, 502, 503, 504):
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def list_cards(session: requests.Session) -> list[dict]:
    """Every English card, as {id, name, set_id, set_name, image}."""
    sets = get_json(session, f"{API}/sets")
    if not sets:
        raise RuntimeError("could not list sets from TCGdex")
    print(f"{len(sets)} English sets")

    cards: list[dict] = []
    seen: set[str] = set()

    def fetch_set(entry: dict):
        return entry, get_json(session, f"{API}/sets/{entry['id']}")

    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, (entry, detail) in enumerate(pool.map(fetch_set, sets), start=1):
            if not detail:
                print(f"  ! set {entry['id']} failed", file=sys.stderr)
                continue
            for card in detail.get("cards", []):
                card_id = card.get("id")
                # Some promos are listed in more than one set.
                if not card_id or not card.get("image") or card_id in seen:
                    continue
                seen.add(card_id)
                cards.append(
                    {
                        "id": card_id,
                        "name": card.get("name", ""),
                        "local_id": str(card.get("localId", "")),
                        "set_id": detail.get("id", entry["id"]),
                        "set_name": detail.get("name", entry.get("name", "")),
                        "image": card["image"],
                    }
                )
            if index % 25 == 0 or index == len(sets):
                print(f"  sets {index}/{len(sets)}  cards so far {len(cards)}")

    cards.sort(key=lambda c: c["id"])
    return cards


def add_dex_ids(session: requests.Session, cards: list[dict], workers: int) -> None:
    """Fill in national dex numbers, one card detail request each."""

    def fetch(card: dict) -> None:
        detail = get_json(session, f"{API}/cards/{card['id']}", attempts=2)
        if detail:
            card["dex"] = detail.get("dexId") or []
            card["category"] = detail.get("category", "")
            card["rarity"] = detail.get("rarity", "")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch, cards), start=1):
            if index % 1000 == 0 or index == len(cards):
                print(f"  dex {index}/{len(cards)}")


def safe_filename(card_id: str, extension: str) -> str:
    """Card ids may contain URL-escaped punctuation (the Unown ! and ? promos)."""
    stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in card_id)
    return f"{stem}.{extension}"


def download_images(
    session: requests.Session,
    cards: list[dict],
    quality: str,
    extension: str,
    workers: int,
) -> tuple[int, int, int]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "fail": 0}

    def fetch(card: dict) -> str:
        destination = IMAGE_DIR / card["file"]
        # A truncated file from a killed run would otherwise be kept forever.
        if destination.exists() and destination.stat().st_size > 1024:
            return "skip"
        url = f"{card['image']}/{quality}.{extension}"
        for attempt in range(3):
            try:
                response = session.get(url, timeout=60)
                if response.status_code == 200 and len(response.content) > 1024:
                    tmp = destination.with_suffix(destination.suffix + ".part")
                    tmp.write_bytes(response.content)
                    tmp.rename(destination)
                    return "ok"
                if response.status_code == 404:
                    return "fail"
            except requests.RequestException:
                pass
            time.sleep(1.0 * (attempt + 1))
        return "fail"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(fetch, cards), start=1):
            counts[result] += 1
            if index % 500 == 0 or index == len(cards):
                print(
                    f"  images {index}/{len(cards)}  "
                    f"new {counts['ok']}  cached {counts['skip']}  failed {counts['fail']}"
                )

    return counts["ok"], counts["skip"], counts["fail"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", default=DEFAULT_QUALITY, choices=("high", "low"))
    parser.add_argument(
        "--extension", default=DEFAULT_EXTENSION, choices=("webp", "png", "jpg")
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--with-dex",
        action="store_true",
        help="also fetch national dex numbers (one request per card, slow)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N cards (smoke test)"
    )
    args = parser.parse_args()

    TCG_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    cards = list_cards(session)
    if args.limit:
        cards = cards[: args.limit]
    print(f"{len(cards)} unique cards")

    for card in cards:
        card["file"] = safe_filename(card["id"], args.extension)

    if args.with_dex:
        print("fetching national dex numbers...")
        add_dex_ids(session, cards, args.workers)

    print("downloading images...")
    downloaded, cached, failed = download_images(
        session, cards, args.quality, args.extension, args.workers
    )

    kept = [c for c in cards if (IMAGE_DIR / c["file"]).exists()]
    with META_PATH.open("w", encoding="utf-8") as handle:
        for card in kept:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")

    total_bytes = sum((IMAGE_DIR / c["file"]).stat().st_size for c in kept)
    print(
        f"\ndone: {len(kept)} cards on disk "
        f"({downloaded} new, {cached} cached, {failed} failed), "
        f"{total_bytes / 1e9:.2f} GB"
    )
    print(f"metadata -> {META_PATH}")
    print(f"images   -> {IMAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
