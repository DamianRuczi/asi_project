"""Prosty test dryfu danych: porównuje rozkład cech w logu predykcji
(logs/predictions.jsonl) z danymi treningowymi (data/01_raw/).

Metoda: z-score średnich. Dla każdej monitorowanej cechy liczymy
|mean_produkcja - mean_trening| / std_trening. Wartość powyżej progu
oznacza, że dane produkcyjne odjechały od treningowych i model może
wymagać ponownego treningu.

Użycie:
    python scripts/check_drift.py
Kod wyjścia 0 = OK, 1 = wykryto dryf (przydatne w automatyzacji).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "01_raw" / "hotel_bookings.csv"
PREDICTIONS_LOG = ROOT / "logs" / "predictions.jsonl"

MONITORED = ["lead_time", "adr", "total_nights"]
Z_THRESHOLD = 0.5  # umowny próg ostrzegawczy


def main() -> int:
    if not PREDICTIONS_LOG.exists():
        print(f"Brak logu predykcji: {PREDICTIONS_LOG} - najpierw wykonaj kilka zadan /predict.")
        return 0

    train = pd.read_csv(TRAIN_CSV)
    train["total_nights"] = train["stays_in_weekend_nights"] + train["stays_in_week_nights"]

    prod = pd.DataFrame(
        json.loads(line) for line in PREDICTIONS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    print(f"Predykcji w logu: {len(prod)}\n")
    print(f"{'cecha':<14}{'trening':>12}{'produkcja':>12}{'z-score':>10}  status")
    print("-" * 56)

    drift = False
    for col in MONITORED:
        mu_t, sd_t = train[col].mean(), train[col].std()
        mu_p = prod[col].mean()
        z = abs(mu_p - mu_t) / sd_t if sd_t else 0.0
        status = "DRYF!" if z > Z_THRESHOLD else "ok"
        drift = drift or z > Z_THRESHOLD
        print(f"{col:<14}{mu_t:>12.2f}{mu_p:>12.2f}{z:>10.2f}  {status}")

    print("\nWniosek:", "wykryto dryf danych - rozwaz ponowny trening." if drift else "rozklady zgodne z treningiem.")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
