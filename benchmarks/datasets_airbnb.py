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

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "datasets", "airbnb_listings_discretized.csv")
OUTCOME = "review_scores_rating"


def airbnb():
    if not os.path.exists(PATH):
        raise SystemExit(f"Manca {PATH}: scaricalo dal link in cima a datasets_airbnb.py")
    df = pd.read_csv(PATH)
    attrs = [c for c in df.columns if c != OUTCOME]
    return df, attrs, OUTCOME, "num", "airbnb"
