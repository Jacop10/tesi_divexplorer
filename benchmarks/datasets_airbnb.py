"""Airbnb New York: annunci di Inside Airbnb, gia' discretizzati dalla prof.

File: datasets/airbnb_listings_discretized.csv, da
https://raw.githubusercontent.com/elianap/divexplorer-generalized/refs/heads/main/data/airbnb_listings_discretized.csv
(preprocessing: notebook D01b-airbnb_processing.ipynb dello stesso repository).

- 26.001 annunci con voto delle recensioni disponibile;
- 24 attributi discretizzati (quartiere, tipo di stanza e di alloggio, prezzo, ospiti,
  bagni, letti, numero di recensioni, superhost, servizi, anni di attivita' dell'host, ...);
- outcome NUMERICO: review_scores_rating, il voto medio delle recensioni (0-5);
- alcuni attributi hanno valori mancanti (fino al 39%): come in DivExplorer, un valore
  mancante non e' un item.

airbnb_v2: lo stesso dataset con il NOSTRO preprocessing, fatto da zero dal file
originale di Inside Airbnb (vedi preprocess_airbnb.py): 17.409 annunci con almeno 5
recensioni, 74 attributi (31 dell'annuncio e dell'host + 43 servizi si/no). Con
n_attrs si tengono i primi n attributi in ordine di entropia decrescente (come per
ACS, Census KDD, Diabetes e HMDA).

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "datasets", "airbnb_listings_discretized.csv")
PATH_V2 = os.path.join(ROOT, "datasets", "airbnb_v2.csv")
OUTCOME = "review_scores_rating"


def airbnb():
    if not os.path.exists(PATH):
        raise SystemExit(f"Manca {PATH}: scaricalo dal link in cima a datasets_airbnb.py")
    df = pd.read_csv(PATH)
    attrs = [c for c in df.columns if c != OUTCOME]
    return df, attrs, OUTCOME, "num", "airbnb"


def airbnb_v2(n_attrs=None):
    if not os.path.exists(PATH_V2):
        raise SystemExit(f"Manca {PATH_V2}: lancia prima python3 benchmarks/preprocess_airbnb.py")
    df = pd.read_csv(PATH_V2, dtype=str, keep_default_na=False)
    df[OUTCOME] = df[OUTCOME].astype(float)

    def entropy(c):
        p = df[c].value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())
    attrs = sorted((c for c in df.columns if c != OUTCOME), key=entropy, reverse=True)
    if n_attrs is not None:
        attrs = attrs[:n_attrs]
    tag = "all" if n_attrs is None else n_attrs
    return df[attrs + [OUTCOME]], attrs, OUTCOME, "num", f"airbnb2_A{tag}"
