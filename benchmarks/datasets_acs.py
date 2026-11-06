"""Dataset reale con tanti attributi: censimento USA (ACS) tramite folktables.

Si usano i dati ACS "grezzi" (centinaia di colonne), non i task gia' pronti di
folktables (che hanno solo ~10 attributi):
- persone adulte (eta' >= 18) con reddito noto;
- target: reddito annuo > 50.000 $;
- un classificatore (random forest, o regressione logistica) addestrato in
  cross-validation a 5 fold, quindi ogni predizione e' fatta su dati non visti;
- outcome "fp" come in DivExplorer: 1 = falso positivo, 0 = vero negativo,
  NaN = classe reale positiva (non conta per l'FPR).

Attributi:
- sempre presenti, discretizzati a mano: eta' (AGE), ore lavorate a settimana
  (HOURS), titolo di studio (EDU);
- poi, fino a n_attrs, le colonne categoriche grezze con 2..max_card valori e
  meno del 50% di mancanti, in ordine di entropia decrescente (le piu'
  "informative"). Si escludono identificativi, pesi, area geografica, flag di
  imputazione e tutte le colonne di reddito (che rivelerebbero il target).
I valori mancanti diventano il valore "NA" (un item come gli altri).

Il dataset preparato viene salvato in benchmarks/data_cache/ e riusato.
Requisiti: pip install folktables scikit-learn
"""
import os

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# identificativi, pesi, geografia, redditi (il target e i suoi componenti)
_EXCLUDE_PREFIX = ("PWGTP", "SERIALNO", "SPORDER", "RT", "ST", "PUMA", "POWPUMA", "POWSP",
                   "MIGPUMA", "MIGSP", "ADJINC", "PINCP", "WAGP", "PERNP", "INTP", "OIP",
                   "PAP", "RETP", "SEMP", "SSIP", "SSP", "POVPIP", "DIVISION", "REGION",
                   "AGEP", "WKHP", "SCHL")      # gli ultimi tre sono aggiunti discretizzati


def _fp_outcome(y_true, y_pred):
    return np.where(y_true == 0, (y_pred == 1).astype(float), np.nan)


def _age_bins(a):
    return pd.cut(a, [17, 24, 34, 44, 54, 64, 200],
                  labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)


def _hours_bins(h):
    out = pd.cut(h.fillna(0), [-1, 0, 19, 34, 40, 49, 200],
                 labels=["0", "1-19", "20-34", "35-40", "41-49", "50+"]).astype(str)
    return out.where(h.notna(), "NA")


def _edu_bins(s):
    # SCHL: 1-15 senza diploma, 16-17 diploma, 18-20 college/associate, 21 bachelor, 22-24 oltre
    out = pd.cut(s.fillna(0), [-1, 15, 17, 20, 21, 24],
                 labels=["no-diploma", "diploma", "college", "bachelor", "master+"]).astype(str)
    return out.where(s.notna(), "NA")


def _value_to_str(x):
    if pd.isna(x):
        return "NA"
    f = float(x)
    return str(int(f)) if f.is_integer() else str(x)


def load_acs_wide(state="CA", year="2018", n_attrs=20, max_card=8, n_rows=100_000,
                  classifier="rf", seed=0):
    """Restituisce (df, attributi, nome). df ha gli attributi (stringhe) e la colonna fp."""
    name = f"acs_{state}_{year}_A{n_attrs}_card{max_card}_n{n_rows}_{classifier}_seed{seed}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"Dataset ACS dalla cache: {cache}")
        return df, [c for c in df.columns if c != "fp"], name

    try:
        from folktables import ACSDataSource
    except ImportError:
        raise SystemExit("Serve folktables: python3 -m pip install folktables scikit-learn")
    print(f"Scarico/carico i dati ACS {state} {year} (la prima volta puo' richiedere qualche minuto)...")
    src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                        root_dir=os.path.join(CACHE_DIR, "folktables"))
    raw = src.get_data(states=[state], download=True)
    raw = raw[(raw["AGEP"] >= 18) & raw["PINCP"].notna()].reset_index(drop=True)
    if n_rows and n_rows < len(raw):
        raw = raw.sample(n_rows, random_state=seed).reset_index(drop=True)
    y = (raw["PINCP"] > 50_000).astype(int).to_numpy()

    # attributi fissi, discretizzati a mano
    disc = pd.DataFrame({"AGE": _age_bins(raw["AGEP"]),
                         "HOURS": _hours_bins(raw["WKHP"]),
                         "EDU": _edu_bins(raw["SCHL"])})

    # altri attributi: categorici grezzi in ordine di entropia
    candidates = []
    for c in raw.columns:
        if c.startswith(_EXCLUDE_PREFIX):
            continue
        if c.startswith("F") and c.endswith("P"):          # flag di imputazione (FAGEP, ...)
            continue
        s = raw[c]
        if s.isna().mean() > 0.5:
            continue
        k = s.nunique(dropna=True)
        if 2 <= k <= max_card:
            freq = s.fillna(-1).value_counts(normalize=True).to_numpy()
            candidates.append((float(-(freq * np.log2(freq)).sum()), c))
    candidates.sort(reverse=True)
    extra = [c for _, c in candidates[:max(0, n_attrs - disc.shape[1])]]
    for c in extra:
        disc[c] = raw[c].map(_value_to_str)
    attrs = list(disc.columns)
    print(f"  {len(raw)} persone, {len(candidates)} colonne candidate, {len(attrs)} attributi:")
    print(f"  {attrs}")

    # classificatore in cross-validation
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    feats = pd.get_dummies(disc, dtype=float)
    feats["AGEP_num"] = raw["AGEP"].to_numpy()
    feats["WKHP_num"] = raw["WKHP"].fillna(0).to_numpy()
    feats["SCHL_num"] = raw["SCHL"].fillna(0).to_numpy()
    model = (RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1,
                                    random_state=seed)
             if classifier == "rf" else LogisticRegression(max_iter=2000))
    print(f"  addestramento del classificatore ({classifier}, 5 fold)...", flush=True)
    y_pred = cross_val_predict(model, feats, y, cv=5)
    print(f"  accuratezza {np.mean(y_pred == y):.3f}, "
          f"FPR {np.mean(y_pred[y == 0] == 1):.3f}, positivi reali {y.mean():.3f}")

    disc["fp"] = _fp_outcome(y, y_pred)
    disc.to_pickle(cache)
    return disc, attrs, name