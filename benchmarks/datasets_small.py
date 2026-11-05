"""Dataset piccoli per i primi confronti.

I file CSV sono quelli della cartella datasets/ del repository di DivExplorer.

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")


def compas(outcome="fpr"):
    """COMPAS: 6.172 persone, 6 attributi. Outcome FPR o FNR del classificatore."""
    from divexplorer.outcomes import (get_false_negative_rate_outcome,
                                      get_false_positive_rate_outcome)
    df = pd.read_csv(os.path.join(DATA, "compas_discretized.csv"))
    attrs = ["age", "charge", "race", "sex", "#prior", "stay"]
    if outcome == "fpr":
        df["fp"] = get_false_positive_rate_outcome(df["class"], df["predicted"])
        return df[attrs + ["fp"]], attrs, "fp", "bool", "compas_fpr"
    df["fn"] = get_false_negative_rate_outcome(df["class"], df["predicted"])
    return df[attrs + ["fn"]], attrs, "fn", "bool", "compas_fnr"


def census():
    """Census: 146.133 persone, 4 attributi. Outcome numerico: reddito (PTOTVAL)."""
    df = pd.read_csv(os.path.join(DATA, "census_income.csv"))
    df["AGE_RANGE"] = 10 * (df["A_AGE"] // 10)
    attrs = ["STATE", "SEX", "EDUCATION", "AGE_RANGE"]
    return df[attrs + ["PTOTVAL"]], attrs, "PTOTVAL", "num", "census"


def synthetic(n=20_000, n_attrs=10, seed=0):
    """Sintetico piccolo: attributi A0..A9 con 3 valori (v0 piu' frequente); due gruppi
    nascosti di 3 attributi in cui il modello fa piu' falsi positivi quando valgono
    tutti v0."""
    rng = np.random.default_rng(seed)
    p = np.exp(-0.7 * np.arange(3))
    p /= p.sum()
    vals = rng.choice(3, size=(n, n_attrs), p=p)
    hidden = [[0, 3, 7], [2, 5, 8]]
    hit = sum(np.all(vals[:, g] == 0, axis=1) for g in hidden)
    y_true = rng.integers(0, 2, n)
    fp_prob = np.clip(0.15 + 0.4 * hit, 0, 1)
    y_pred = np.where(y_true == 0, rng.random(n) < fp_prob, rng.random(n) < 0.8)
    df = pd.DataFrame({f"A{i}": np.char.add("v", vals[:, i].astype(str))
                       for i in range(n_attrs)})
    df["fp"] = np.where(y_true == 0, y_pred.astype(float), np.nan)
    return df, [f"A{i}" for i in range(n_attrs)], "fp", "bool", "synthetic_small"


LOADERS = {
    "compas_fpr": lambda: compas("fpr"),
    "compas_fnr": lambda: compas("fnr"),
    "census": census,
    "synthetic": synthetic,
}
