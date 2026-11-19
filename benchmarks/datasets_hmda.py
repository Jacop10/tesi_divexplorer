"""HMDA (Home Mortgage Disclosure Act): richieste di mutuo negli Stati Uniti.

Dati pubblici del CFPB (Consumer Financial Protection Bureau), un file per stato e
anno, scaricato con la Data Browser API. Per lo stato di New York, anno 2022, solo
mutui concessi (action_taken = 1) e rifiutati (action_taken = 3):

  mkdir -p datasets/hmda
  curl -L "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=NY&years=2022&actions_taken=1,3" -o datasets/hmda/hmda_NY_2022.csv

Documentazione dei campi:
https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/public-lar-schema

Diverso dagli altri dataset: l'outcome NON e' l'errore di un classificatore, ma la
decisione reale della banca:
- outcome "denied": 1 = mutuo rifiutato, 0 = mutuo concesso. La divergenza di un
  sottogruppo e' quindi quanto il suo tasso di rifiuto si discosta da quello medio.

Preparazione:
- si tolgono i campi che esistono solo per i mutui concessi o che rivelano l'esito
  (motivi del rifiuto, tasso di interesse, costi, commissioni, a chi e' stato venduto
  il mutuo, ...), gli identificativi (banca, sezione di censimento), i campi
  dettagliati di etnia e razza (restano le versioni "derived_") e i campi ripetuti
  (aus_2..5, ...);
- attributi numerici (importo, valore dell'immobile, reddito, rapporto prestito/valore,
  durata, caratteristiche della zona) in 3 fasce con lo stesso numero di richieste;
  il rapporto debito/reddito e' gia' a fasce (i valori singoli 36-49 diventano "36-49");
- valori presenti in meno dell'1% delle richieste -> "other"; mancanti, "NA", "Exempt"
  e codici di "non disponibile" restano come categorie;
- attributi: sempre razza, sesso ed eta' del richiedente, poi fino a n_attrs gli altri
  in ordine di entropia decrescente (come per ACS).
Il dataset preparato viene salvato in benchmarks/data_cache/ e riusato.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "datasets", "hmda")
CACHE_DIR = os.path.join(HERE, "data_cache")

FIXED = ["derived_race", "derived_sex", "applicant_age"]
# campi da togliere: esito o solo per i concessi, identificativi, duplicati
DROP = (["activity_year", "lei", "state_code", "census_tract", "action_taken",
         "purchaser_type", "interest_rate", "rate_spread", "hoepa_status",
         "total_loan_costs", "total_points_and_fees", "origination_charges",
         "discount_points", "lender_credits", "prepayment_penalty_term",
         "intro_rate_period", "multifamily_affordable_units",
         "applicant_age_above_62", "co_applicant_age_above_62",
         "applicant_ethnicity_observed", "co_applicant_ethnicity_observed",
         "applicant_race_observed", "co_applicant_race_observed",
         "applicant_sex_observed", "co_applicant_sex_observed"]
        + [f"denial_reason_{i}" for i in range(1, 5)]
        + [f"aus_{i}" for i in range(2, 6)]
        + [f"{who}_{what}_{i}" for who in ("applicant", "co_applicant")
           for what in ("ethnicity", "race") for i in range(1, 6)])
NUMERIC = ["loan_amount", "combined_loan_to_value_ratio", "loan_to_value_ratio",
           "loan_term", "property_value",
           "income", "tract_population", "tract_minority_population_percent",
           "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
           "tract_owner_occupied_units", "tract_one_to_four_family_homes",
           "tract_median_age_of_housing_units"]


def _terciles(s):
    """Tre fasce con lo stesso numero di richieste; i valori non numerici ("NA",
    "Exempt", vuoti) restano come categoria."""
    x = pd.to_numeric(s, errors="coerce")
    out = s.astype(str).where(s.notna(), "NA").astype(object)
    ok = x.notna()
    if ok.sum() >= 3:
        q = np.unique(np.quantile(x[ok], [1 / 3, 2 / 3]))
        edges = [-np.inf] + [float(v) for v in q] + [np.inf]
        labels = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            labels.append(f"<={hi:g}" if i == 0 else (f">{lo:g}" if not np.isfinite(hi)
                                                        else f"({lo:g}-{hi:g}]"))
        out[ok] = pd.cut(x[ok], edges, labels=labels).astype(str)
    return out


def _dti(s):
    """Rapporto debito/reddito: i valori singoli 36..49 diventano la fascia "36-49"."""
    s = s.astype(str).where(s.notna(), "NA")
    num = pd.to_numeric(s, errors="coerce")
    return s.where(~num.between(36, 49), "36-49")


def _prepare(raw):
    raw = raw[raw["action_taken"].isin([1, 3])].reset_index(drop=True)
    y = (raw["action_taken"] == 3).astype(float).to_numpy()
    df = raw.drop(columns=[c for c in DROP if c in raw.columns])
    disc = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in NUMERIC:
            disc[c] = _terciles(df[c])
        elif c == "debt_to_income_ratio":
            disc[c] = _dti(df[c])
        else:
            disc[c] = df[c].astype(str).where(df[c].notna(), "NA")
        freq = disc[c].value_counts(normalize=True)
        disc[c] = disc[c].where(~disc[c].isin(freq[freq < 0.01].index), "other")
    keep = [c for c in disc.columns if disc[c].value_counts(normalize=True).iloc[0] < 0.995]
    leak = [c for c in keep if c.startswith(("denial_reason", "interest_rate", "rate_spread"))]
    assert not leak, f"Colonne che rivelano l'esito ancora presenti: {leak}"
    return disc[keep], y


def load_hmda(state="NY", year=2022, n_attrs=None, n_rows=None, seed=0):
    """Restituisce (df, attributi, nome). df ha gli attributi (stringhe) e la colonna
    denied. n_attrs=None: tutti; n_rows=None: tutte le richieste."""
    tag = "all" if n_attrs is None else n_attrs
    name = f"hmda_v2_{state}_{year}_A{tag}_n{n_rows}_seed{seed}"   # v2: nomi con trattino
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"HMDA dalla cache: {cache}")
        return df, [c for c in df.columns if c != "denied"], name
    path = os.path.join(DATA_DIR, f"hmda_{state}_{year}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Manca {path}: vedi le istruzioni in cima a datasets_hmda.py")

    raw = pd.read_csv(path, low_memory=False, dtype=str, keep_default_na=False,
                      na_values=[""])
    # nei file scaricati i nomi usano il trattino ("aus-1", "co-applicant_age",
    # "denial_reason-1", "derived_msa-md"): si uniformano al trattino basso, come nella
    # documentazione, cosi' le colonne da togliere vengono riconosciute
    raw.columns = [c.replace("-", "_") for c in raw.columns]
    raw["action_taken"] = pd.to_numeric(raw["action_taken"], errors="coerce")
    if n_rows and n_rows < len(raw):
        raw = raw.sample(n_rows, random_state=seed).reset_index(drop=True)
    disc, y = _prepare(raw)

    def entropy(s):
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())
    fixed = [c for c in FIXED if c in disc.columns]
    others = sorted((c for c in disc.columns if c not in fixed),
                    key=lambda c: entropy(disc[c]), reverse=True)
    if n_attrs is not None:
        others = others[:max(0, n_attrs - len(fixed))]
    attrs = fixed + others
    disc = disc[attrs].copy()
    disc["denied"] = y
    print(f"  {len(disc)} richieste, tasso di rifiuto {y.mean():.3f}, "
          f"{len(attrs)} attributi: {attrs}")
    disc.to_pickle(cache)
    return disc, attrs, name


def hmda(n_attrs=None, state="NY", year=2022):
    df, attrs, _ = load_hmda(state=state, year=year, n_attrs=n_attrs)
    return df, attrs, "denied", "bool", f"hmda_{state}{year}_A{'all' if n_attrs is None else n_attrs}"
