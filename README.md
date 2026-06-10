# Predykcja anulowania rezerwacji hotelowych

Projekt zaliczeniowy **ASI** — kompletny pipeline **ML / MLOps end-to-end**:
od eksploracji danych, przez potok Kedro i strojenie modelu, po wdrożenie modelu
jako API z monitoringiem, uruchamiane w Dockerze.

**Zespół:** Jastrzębski, Ruczyński

---

## Problem

Hotele tracą przychód i moce przerobowe z powodu rezerwacji, które zostają
**anulowane**. Gdyby ryzyko anulowania dało się oszacować **już w momencie
rezerwacji**, hotel mógłby reagować (polityka depozytów, kontrolowany overbooking,
oferty dla gości zagrożonych rezygnacją), ograniczając straty.

Budujemy model **klasyfikacji binarnej** przewidujący, czy dana rezerwacja
zostanie anulowana (`is_canceled`).

## Dane

- **Hotel Booking Demand** (Antonio, Almeida & Nunes, 2019) — realne rezerwacje
  dwóch portugalskich hoteli (miejski i resort) z lat **2015–2017**.
- **119 390** rezerwacji, **32** kolumny.
- Zmienna celu: **`is_canceled`** (1 = anulowana, 0 = zrealizowana); ok. **37%**
  rezerwacji anulowanych.
- Źródło: [Kaggle — jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
  Pobranie bez logowania: `python scripts/fetch_data.py`.

> **Uwaga (data leakage):** kolumny `reservation_status` i
> `reservation_status_date` opisują *wynik* rezerwacji i są znane dopiero po
> fakcie — są celowo usuwane z cech, aby model był użyteczny w praktyce.

## Architektura docelowa

```
 data/01_raw/hotel_bookings.csv
            │
            ▼
 ┌─────────────────────── Kedro ───────────────────────┐
 │  pipeline: data_processing   →   pipeline: data_science │
 │  (czyszczenie, cechy,            (split, trening,        │
 │   kodowanie)                      ewaluacja)             │
 └──────────────────────────────────────────────────────┘
            │                         │
            │ (MLflow: śledzenie       │ (Optuna: strojenie,
            │  eksperymentów)          │  AutoGluon: AutoML)
            ▼                         ▼
   DVC (wersjonowanie danych)   MLflow Model Registry (najlepszy model)
            │
            ▼
 ┌──────────────── Docker Compose ────────────────┐
 │  FastAPI  /predict   ──►  Prometheus  /metrics   │
 │  (serwowanie modelu)      (monitoring predykcji) │
 └─────────────────────────────────────────────────┘
```

Diagram w wersji graficznej (draw.io) znajdzie się w `docs/`.

## Stos technologiczny

| Obszar | Narzędzie |
|---|---|
| Język / analiza | Python, pandas, NumPy, scikit-learn |
| EDA / wizualizacje | matplotlib, seaborn |
| Pipeline ML | **Kedro** (+ kedro-viz) |
| Śledzenie eksperymentów | **MLflow** |
| AutoML | **AutoGluon** |
| Strojenie hiperparametrów | **Optuna** (optymalizacja bayesowska) |
| Modele | Random Forest, XGBoost, regresja logistyczna |
| Serwowanie | **FastAPI** + Uvicorn |
| Monitoring | **Prometheus** (+ Grafana opcjonalnie) |
| Konteneryzacja | **Docker** + Docker Compose |
| MLOps (ścieżka A) | **DVC** (dane) + **MLflow Model Registry** |
| CI | GitHub Actions (ruff + pytest) |

## Struktura repozytorium

```
asi_project/
├── conf/
│   └── base/
│       ├── catalog.yml         # definicje zbiorów danych (warstwy 01_raw…08_reporting)
│       └── parameters.yml      # parametry potoku i modelu
├── data/                       # warstwy danych wg konwencji Kedro (poza gitem)
├── docs/                       # dokumentacja PDF + diagram architektury
├── notebooks/
│   └── 01_baseline.ipynb       # wersja podstawowa: EDA + preprocessing + baseline
├── scripts/
│   └── fetch_data.py           # pobranie surowych danych
├── src/asi_project/
│   └── pipelines/
│       ├── data_processing/    # czyszczenie -> inżynieria cech -> kodowanie
│       └── data_science/       # split -> trening -> ewaluacja (+ MLflow)
├── tests/                      # testy jednostkowe węzłów (pytest)
├── pyproject.toml              # konfiguracja Kedro, ruff, pytest
├── requirements.txt
└── README.md
```

## Uruchomienie

### 1. Środowisko
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dane
```bash
python scripts/fetch_data.py
```

### 3. Notebook (wersja podstawowa)
```bash
jupyter notebook notebooks/01_baseline.ipynb
```

### 4. Potok Kedro (przetwarzanie + trening + ewaluacja)
```bash
kedro run                      # cały potok end-to-end
kedro run --pipeline data_science   # tylko część modelowa
kedro viz                      # interaktywny graf potoku w przeglądarce
```

### 5. Śledzenie eksperymentów (MLflow)
Każde `kedro run` zapisuje parametry, metryki i artefakt modelu jako run MLflow.
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # podgląd: http://127.0.0.1:5000
```

### 6. Testy i jakość kodu
```bash
pytest          # testy jednostkowe węzłów
ruff check src tests
```

> Kolejne etapy (serwis `docker compose up`) zostaną dodane wraz z rozwojem
> projektu — patrz roadmapa niżej.

## Wyniki

| Etap | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (notebook) | 0.869 | 0.811 | 0.933 |
| Potok Kedro (po inżynierii cech) | **0.890** | **0.846** | **0.956** |

Inżynieria cech (`total_nights`, `total_guests`, `is_family`, `room_changed`,
`has_agent`, grupowanie `country`) mierzalnie poprawiła wszystkie metryki.
Najsilniejsze predyktory anulowania: `lead_time`, `deposit_type = Non Refund`,
`adr`, liczba próśb specjalnych, wcześniejsze anulowania.

## Roadmapa

- [x] **Wersja podstawowa** — EDA, preprocessing, model bazowy (RF), ewaluacja
- [x] **Pipeline Kedro** — `data_processing` + `data_science` (+ pytest, ruff)
- [x] **Śledzenie eksperymentów** — MLflow (parametry, metryki, artefakt modelu)
- [x] **Inżynieria cech** — nowe cechy domenowe w potoku (wzrost wszystkich metryk)
- [ ] **Udoskonalanie** — porównanie modeli (RF/XGBoost/LogReg), Optuna, AutoGluon
- [ ] **Produkcja** — FastAPI + Prometheus + Docker Compose, monitoring + drift
- [ ] **MLOps (A)** — DVC + MLflow Model Registry (+ CI: GitHub Actions)
- [ ] **Dokumentacja** — PDF + diagram architektury
- [ ] **Prezentacja** — slajdy 10–15 min + demo
