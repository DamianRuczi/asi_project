"""Potok 'data_processing' — czyszczenie danych i inżynieria cech.

Wejście:  surowe rezerwacje (data/01_raw/hotel_bookings.csv)
Wyjście:  zakodowana macierz cech gotowa do modelowania (model_input)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Kolumny opisujące WYNIK rezerwacji — znane dopiero po fakcie.
# Pozostawienie ich = wyciek informacji (data leakage) i model bezużyteczny
# w praktyce, dlatego usuwamy je na samym początku.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]


def clean_data(hotels: pd.DataFrame) -> pd.DataFrame:
    """Czyszczenie surowych danych: leakage, braki, oczywiste błędy."""
    df = hotels.copy()

    df = df.drop(columns=LEAKAGE_COLUMNS)
    df = df.drop(columns=["company"])  # ~94% braków — kolumna nieużyteczna

    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("UNKNOWN")
    df["adr"] = df["adr"].clip(lower=0)  # pojedyncza ujemna cena za dobę

    logger.info("clean_data: %s wierszy, %s kolumn", *df.shape)
    return df


def engineer_features(df: pd.DataFrame, top_n_countries: int) -> pd.DataFrame:
    """Nowe cechy domenowe + redukcja kardynalności country/agent."""
    df = df.copy()

    # cechy domenowe
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["room_changed"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)

    # agent: identyfikator o dużej kardynalności -> binarna informacja
    # "czy rezerwacja przez pośrednika" (NaN = rezerwacja bezpośrednia)
    df["has_agent"] = df["agent"].notna().astype(int)
    df = df.drop(columns=["agent"])

    # country: ~180 wartości -> top N + "OTHER" (czytelne one-hot)
    top = df["country"].value_counts().nlargest(top_n_countries).index
    df["country"] = df["country"].where(df["country"].isin(top), "OTHER")

    # błąd danych: rezerwacje bez ani jednego gościa
    before = len(df)
    df = df[df["total_guests"] > 0]
    logger.info(
        "engineer_features: usunieto %s rezerwacji bez gosci", before - len(df)
    )
    return df


def encode_features(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encoding zmiennych kategorycznych.

    Zwraca też listę kolumn-cech — potrzebną później przy serwowaniu modelu
    (API musi ułożyć wejście w identycznym porządku kolumn).
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    features = [c for c in encoded.columns if c != target_column]
    logger.info("encode_features: %s cech po kodowaniu", len(features))
    return encoded, features
