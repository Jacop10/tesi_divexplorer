"""Dataset medi: qui DivExplorer comincia a rallentare, ma finisce ancora.

- synthetic15, synthetic20: lo stesso generatore di datasets_small (attributi con 3
  valori, due gruppi nascosti con piu' falsi positivi), 50.000 righe;
- acs15, acs20: censimento USA (California 2018, 100.000 adulti) con un random
  forest; outcome = falsi positivi (FPR). Serve datasets_acs.py e
  pip install folktables scikit-learn. La prima volta scarica i dati (qualche
  minuto), poi li riusa da benchmarks/data_cache/.

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
from datasets_small import synthetic


def synthetic_medium(n_attrs, n=50_000, seed=0):
    df, attrs, outcome, kind, _ = synthetic(n=n, n_attrs=n_attrs, seed=seed)
    return df, attrs, outcome, kind, f"synthetic_A{n_attrs}"


def acs(n_attrs):
    from datasets_acs import load_acs_wide
    df, attrs, _ = load_acs_wide(n_attrs=n_attrs)
    return df, attrs, "fp", "bool", f"acs_A{n_attrs}"


LOADERS = {
    "synthetic15": lambda: synthetic_medium(15),
    "synthetic20": lambda: synthetic_medium(20),
    "acs15": lambda: acs(15),
    "acs20": lambda: acs(20),
}
