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
├── app/
│   ├── main.py                 # FastAPI: /predict, /health, /metrics
│   └── preprocessing.py        # przygotowanie cech 1:1 z potokiem treningowym
├── monitoring/
│   └── prometheus.yml          # konfiguracja scrape'owania metryk API
├── src/asi_project/
│   └── pipelines/
│       ├── data_processing/    # czyszczenie -> inżynieria cech -> kodowanie
│       └── data_science/       # split -> trening -> ewaluacja (+ MLflow)
├── tests/                      # testy jednostkowe węzłów (pytest)
├── Dockerfile                  # obraz serwujący (model przez wolumen)
├── docker-compose.yml          # API + Prometheus
├── sample_booking.json         # przykładowe żądanie do /predict
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

### 7. API predykcyjne (lokalnie)
```bash
uvicorn app.main:app --reload
```
Interaktywna dokumentacja z wbudowanym przykładem: http://127.0.0.1:8000/docs
Przykładowe żądania (PowerShell):
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body (Get-Content sample_booking.json -Raw)
```

### 8. Produkcja: Docker + Prometheus
```bash
kedro run                  # najpierw artefakty modelu (montowane jako wolumen)
docker compose up --build  # API: http://127.0.0.1:8000/docs · Prometheus: http://127.0.0.1:9090
```
Model **nie jest częścią obrazu** — kontener montuje `data/06_models/` jako
wolumen (podmiana modelu po retreningu bez przebudowy obrazu), a `.dockerignore`
trzyma kontekst builda z dala od danych i środowisk. Metryki API (liczniki
predykcji wg klasy, latencja, błędy) zbiera Prometheus (target `api:8000/metrics`).

### 9. Logowanie predykcji i test dryfu danych
Każde żądanie `/predict` jest dopisywane do `logs/predictions.jsonl`. Prosty
test dryfu porównuje średnie kluczowych cech (lead_time, adr, total_nights)
z danymi treningowymi metodą z-score:
```bash
python scripts/check_drift.py
```

## Wyniki

| Etap | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Baseline (notebook) | 0.869 | 0.811 | 0.933 |
| Potok Kedro (po inżynierii cech) | **0.890** | **0.846** | **0.956** |

Inżynieria cech (`total_nights`, `total_guests`, `is_family`, `room_changed`,
`has_agent`, grupowanie `country`) mierzalnie poprawiła wszystkie metryki.
Najsilniejsze predyktory anulowania: `lead_time`, `deposit_type = Non Refund`,
`adr`, liczba próśb specjalnych, wcześniejsze anulowania.

### Porównanie modeli

Modele są porównywane przez stratyfikowaną 3-krotną walidację krzyżową wyłącznie
na zbiorze treningowym. Zbiór testowy pozostaje nietknięty do końcowej oceny.
Metryką wyboru jest ROC-AUC.

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8175 | 0.7301 | 0.8984 |
| Random Forest | **0.8853** | **0.8380** | **0.9529** |
| XGBoost | 0.8672 | 0.8138 | 0.9427 |

Random Forest wygrał porównanie i po treningu na pełnym zbiorze treningowym
osiągnął na zbiorze testowym `accuracy=0.8901`, `F1=0.8457`,
`ROC-AUC=0.9561`. Wyniki kandydatów i modelu końcowego są śledzone w MLflow.

### Strojenie Optuną

Optuna stroi zwycięski Random Forest przez 3-krotną walidację krzyżową wyłącznie
na zbiorze treningowym. Budżet jest ograniczony do 15 prób lub 600 sekund.
W wykonanym przebiegu limit czasu pozwolił ukończyć 13 prób.

Najlepsze parametry:

```yaml
n_estimators: 400
max_depth: null
min_samples_split: 6
min_samples_leaf: 1
max_features: 0.7
```

| Random Forest | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Przed strojeniem | **0.8901** | 0.8457 | 0.9561 |
| Po Optunie | 0.8887 | **0.8459** | **0.9564** |

Strojenie dało niewielką poprawę F1 i ROC-AUC kosztem accuracy. Jest to
oczekiwany wynik, ponieważ Optuna optymalizowała ROC-AUC, a model bazowy był już
mocny.

### AutoGluon

Notebook `notebooks/02_autogluon.ipynb` uruchamia AutoML w osobnym środowisku,
aby jego zależności nie zmieniały wersji używanych przez pipeline Kedro. Trening
ma limit 300 sekund i używa presetu `medium_quality`.

```powershell
python -m venv .venv-autogluon
.venv-autogluon\Scripts\python.exe -m pip install -r requirements-autogluon.txt
.venv-autogluon\Scripts\jupyter-nbconvert.exe --execute --inplace notebooks/02_autogluon.ipynb
```

AutoGluon zakończył trening w około minutę. Najlepszym modelem został ensemble
łączący dostępne modele drzewiaste.

| Model AutoGluon | Test ROC-AUC |
|---|---:|
| WeightedEnsemble_L2 | **0.9562** |
| RandomForestEntr | 0.9561 |
| RandomForestGini | 0.9551 |
| ExtraTreesEntr | 0.9520 |
| ExtraTreesGini | 0.9513 |

| Model końcowy | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Random Forest po Optunie | 0.8887 | **0.8459** | **0.9564** |
| AutoGluon WeightedEnsemble | **0.8889** | 0.8438 | 0.9562 |

Wyniki są praktycznie remisowe. Do dalszego wdrożenia wybieramy Random Forest
po Optunie, ponieważ minimalnie wygrywa w F1 i ROC-AUC, a pojedynczy model jest
prostszy do wyjaśnienia, serwowania i monitorowania niż ensemble AutoML.

## Roadmapa

- [x] **Wersja podstawowa** — EDA, preprocessing, model bazowy (RF), ewaluacja
- [x] **Pipeline Kedro** — `data_processing` + `data_science` (+ pytest, ruff)
- [x] **Śledzenie eksperymentów** — MLflow (parametry, metryki, artefakt modelu)
- [x] **Inżynieria cech** — nowe cechy domenowe w potoku (wzrost wszystkich metryk)
- [x] **Porównanie modeli** — RF/XGBoost/LogReg przez CV, wybór po ROC-AUC
- [x] **Strojenie Optuną** — kontrolowany budżet prób, wynik zapisany w MLflow
- [x] **AutoGluon** — wykonany notebook, leaderboard i porównanie z Optuną
- [x] **Produkcja** — FastAPI (`/predict`, `/health`, `/docs`) + Prometheus + Docker Compose; logowanie predykcji + test dryfu (z-score)
- [ ] **MLOps (A)** — DVC + MLflow Model Registry (+ CI: GitHub Actions)
- [ ] **Dokumentacja** — PDF + diagram architektury
- [ ] **Prezentacja** — slajdy 10–15 min + demo
