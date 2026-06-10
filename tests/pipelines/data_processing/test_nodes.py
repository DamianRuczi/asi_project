"""Testy jednostkowe węzłów potoku data_processing."""

import pandas as pd

from asi_project.pipelines.data_processing.nodes import (
    LEAKAGE_COLUMNS,
    clean_data,
    encode_features,
    engineer_features,
)


def _sample_raw() -> pd.DataFrame:
    """Minimalna ramka z kolumnami używanymi przez węzły."""
    return pd.DataFrame(
        {
            "is_canceled": [0, 1, 0],
            "reservation_status": ["Check-Out", "Canceled", "Check-Out"],
            "reservation_status_date": ["2017-01-01"] * 3,
            "company": [None, None, 40.0],
            "children": [1.0, None, 0.0],
            "country": ["PRT", None, "GBR"],
            "adr": [100.0, -5.0, 80.0],
            "stays_in_weekend_nights": [1, 0, 2],
            "stays_in_week_nights": [2, 3, 0],
            "adults": [2, 0, 1],
            "babies": [0, 0, 0],
            "agent": [9.0, None, 14.0],
            "reserved_room_type": ["A", "A", "B"],
            "assigned_room_type": ["A", "C", "B"],
        }
    )


def test_clean_data_usuwa_leakage_i_uzupelnia_braki():
    out = clean_data(_sample_raw())

    for col in LEAKAGE_COLUMNS + ["company"]:
        assert col not in out.columns

    assert out["children"].isna().sum() == 0
    assert (out["adr"] >= 0).all()  # ujemne ceny przycięte do zera
    assert out["country"].isna().sum() == 0


def test_engineer_features_liczy_cechy_i_usuwa_puste_rezerwacje():
    out = engineer_features(clean_data(_sample_raw()), top_n_countries=2)

    # wiersz nr 2 ma 0 gości (adults=0, children=NaN->0, babies=0) -> usunięty
    assert len(out) == 2
    assert "total_nights" in out.columns and "total_guests" in out.columns
    assert out.iloc[0]["total_nights"] == 3  # 1 weekendowa + 2 robocze
    assert "agent" not in out.columns and "has_agent" in out.columns
    assert set(out["room_changed"]) <= {0, 1}


def test_encode_features_zwraca_macierz_numeryczna_i_liste_cech():
    encoded, features = encode_features(
        engineer_features(clean_data(_sample_raw()), top_n_countries=2),
        target_column="is_canceled",
    )

    assert "is_canceled" in encoded.columns
    assert "is_canceled" not in features
    # po one-hot nie zostaje żadna kolumna tekstowa
    assert encoded.select_dtypes(include=["object"]).empty
    assert len(features) == encoded.shape[1] - 1
