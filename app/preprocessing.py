"""Przygotowanie cech do predykcji — lustrzane odbicie potoku treningowego.

API musi wykonać na pojedynczej rezerwacji dokładnie te same przekształcenia,
które potok data_processing wykonuje na całym zbiorze treningowym:
- cechy domenowe (total_nights, total_guests, is_family, room_changed, has_agent),
- redukcja country do kategorii znanych z treningu (pozostałe -> OTHER),
- one-hot encoding i ułożenie kolumn dokładnie w porządku z treningu.

Kolumny dummy, których nie ma w pojedynczym wierszu (inne kategorie oraz
kategorie bazowe po drop_first), uzupełniamy zerami przez reindex.
"""

import pandas as pd


def known_countries(model_features: list[str]) -> set[str]:
    """Kraje zakodowane w treningu jako osobne kolumny (country_XXX)."""
    return {
        c.removeprefix("country_")
        for c in model_features
        if c.startswith("country_")
    }


def prepare_features(booking: dict, model_features: list[str]) -> pd.DataFrame:
    """Zamienia surową rezerwację (dict) na macierz cech zgodną z modelem."""
    df = pd.DataFrame([booking])

    # te same korekty co clean_data (na polach, które istnieją w żądaniu)
    df["children"] = df["children"].fillna(0)
    df["adr"] = df["adr"].clip(lower=0)

    # te same cechy domenowe co engineer_features
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["room_changed"] = (
        df["reserved_room_type"] != df["assigned_room_type"]
    ).astype(int)
    df["has_agent"] = df["agent"].notna().astype(int)
    df = df.drop(columns=["agent"])

    # country: kategorie spoza treningowego top-N -> OTHER (jak w potoku)
    df["country"] = df["country"].fillna("UNKNOWN")
    df["country"] = df["country"].where(
        df["country"].isin(known_countries(model_features)), "OTHER"
    )

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(df, columns=cat_cols)
    return encoded.reindex(columns=model_features, fill_value=0)
