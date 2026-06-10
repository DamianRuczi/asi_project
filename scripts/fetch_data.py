"""Pobiera surowy zbiór Hotel Booking Demand do data/01_raw/.

Zbiór pochodzi z Kaggle (jessemostipak/hotel-booking-demand). Ponieważ Kaggle
wymaga logowania, korzystamy z publicznego, identycznego mirrora w repozytorium
TidyTuesday (raw.githubusercontent.com) — te same dane, bez uwierzytelniania.

Użycie:
    python scripts/fetch_data.py
"""
from pathlib import Path
import sys
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "01_raw" / "hotel_bookings.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Plik już istnieje: {DEST} — pomijam pobieranie.")
        return 0
    print(f"Pobieram dane z:\n  {URL}")
    df = pd.read_csv(URL)
    df.to_csv(DEST, index=False)
    print(f"Zapisano {len(df):,} wierszy × {df.shape[1]} kolumn do:\n  {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
