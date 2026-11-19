"""Diabetes 130-US hospitals (1999-2008): ricoveri di pazienti diabetici.

Dati: UCI Machine Learning Repository, dataset 296,
https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
Scaricare lo zip, estrarre e mettere il file diabetic_data.csv in
datasets/diabetes/diabetic_data.csv.

- 101.766 ricoveri in 130 ospedali USA; una riga per ricovero;
- target: riammissione entro 30 giorni (readmitted = "<30", circa 11%); classificatore
  random forest BILANCIATO in cross-validation a 5 fold (come per Census KDD: la
  classe e' rara e senza bilanciamento il FPR sarebbe quasi 0);
- outcome "fp": 1 = falso positivo, 0 = vero negativo, NaN = classe reale positiva.

Preparazione (come nello studio originale, Strack et al., 2014):
- si tolgono gli identificativi (encounter_id, patient_nbr) e il peso (weight,
  mancante nel 97% dei casi);
- si tolgono i ricoveri finiti con decesso o in hospice (discharge_disposition_id 11,
  13, 14, 19, 20, 21): non possono essere riammessi;
- le diagnosi (diag_1, diag_2, diag_3, codici ICD-9) diventano 9 gruppi: circolatorio,
  respiratorio, digestivo, diabete, lesioni, muscoloscheletrico, genitourinario,
  tumori, altro;
- attributi numerici discretizzati: giorni di ricovero, esami, procedure, farmaci,
  visite precedenti (ambulatoriali, di emergenza, ricoveri), numero di diagnosi;
- valori presenti in meno dell'1% dei ricoveri -> "other", mancanti ("?") -> "NA";
- si tolgono gli attributi quasi costanti (un valore in almeno il 99,5% dei ricoveri,
  ad esempio molti farmaci quasi mai usati): non dicono niente e farebbero solo
  esplodere il numero di sottogruppi frequenti;
- attributi: sempre eta', sesso ed etnia, poi fino a n_attrs gli altri in ordine di
  entropia decrescente (come per ACS e Census KDD).
Il dataset preparato viene salvato in benchmarks/data_cache/ e riusato.
Requisiti: scikit-learn.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATH = os.path.join(ROOT, "datasets", "diabetes", "diabetic_data.csv")
CACHE_DIR = os.path.join(HERE, "data_cache")

FIXED = ["age", "gender", "race"]
DROP = ["encounter_id", "patient_nbr", "weight"]
DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}
NUMERIC = {   # colonna: (limiti degli intervalli, etichette)
    "time_in_hospital": ([0, 2, 4, 7, 99], ["1-2", "3-4", "5-7", "8+"]),
    "num_lab_procedures": ([-1, 30, 50, 999], ["0-30", "31-50", "51+"]),
    "num_procedures": ([-1, 0, 1, 2, 99], ["0", "1", "2", "3+"]),
    "num_medications": ([-1, 10, 20, 999], ["0-10", "11-20", "21+"]),
    "number_outpatient": ([-1, 0, 1, 999], ["0", "1", "2+"]),
    "number_emergency": ([-1, 0, 1, 999], ["0", "1", "2+"]),
    "number_inpatient": ([-1, 0, 1, 999], ["0", "1", "2+"]),
    "number_diagnoses": ([-1, 5, 8, 99], ["1-5", "6-8", "9+"]),
}


def _icd9_group(code):
    """Gruppo di una diagnosi ICD-9 (Strack et al., 2014)."""
    if pd.isna(code):
        return "NA"
    c = str(code)
    if c.startswith(("V", "E")):
        return "other"
    try:
        x = float(c)
    except ValueError:
        return "other"
    if 390 <= x <= 459 or int(x) == 785:
        return "circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "digestive"
    if int(x) == 250:
        return "diabetes"
    if 800 <= x <= 999:
        return "injury"
    if 710 <= x <= 739:
        return "musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "genitourinary"
    if 140 <= x <= 239:
        return "neoplasms"
    return "other"


def _prepare(raw):
    df = raw.drop(columns=[c for c in DROP if c in raw.columns])
    df = df[~df["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)].reset_index(drop=True)
    y = (df["readmitted"].astype(str) == "<30").astype(int).to_numpy()
    num = df[[c for c in NUMERIC if c in df.columns]].astype(float)

    disc = pd.DataFrame(index=df.index)
    for c, (edges, labels) in NUMERIC.items():
        if c in df.columns:
            disc[c] = pd.cut(df[c], edges, labels=labels).astype(str)
    for c in ("diag_1", "diag_2", "diag_3"):
        if c in df.columns:
            disc[c] = df[c].map(_icd9_group)
    for c in df.columns:
        if c in disc.columns or c == "readmitted":
            continue
        s = df[c].astype(str).where(df[c].notna(), "NA")
        freq = s.value_counts(normalize=True)
        disc[c] = s.where(~s.isin(freq[freq < 0.01].index), "other")
    # attributi quasi costanti: fuori
    keep = [c for c in disc.columns if disc[c].value_counts(normalize=True).iloc[0] < 0.995]
    return disc[keep], y, num


def load_diabetes(n_attrs=None, classifier="rf", balanced=True, seed=0):
    """Restituisce (df, attributi, nome). n_attrs=None: tutti gli attributi."""
    tag = "all" if n_attrs is None else n_attrs
    name = f"diabetes_A{tag}_{classifier}{'_bal' if balanced else ''}_seed{seed}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"Diabetes dalla cache: {cache}")
        return df, [c for c in df.columns if c != "fp"], name
    if not os.path.exists(PATH):
        raise SystemExit(f"Manca {PATH}: vedi le istruzioni in cima a datasets_diabetes.py")

    # "?" = mancante; keep_default_na=False perche' "None" e' un valore vero
    # (max_glu_serum, A1Cresult: esame non fatto)
    raw = pd.read_csv(PATH, na_values=["?"], keep_default_na=False, low_memory=False)
    disc, y, num = _prepare(raw)

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
    print(f"  {len(disc)} ricoveri, {len(attrs)} attributi: {attrs}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    feats = pd.get_dummies(disc, dtype=float)
    for c in num.columns:
        feats[f"{c}_num"] = num[c].to_numpy()
    weights = "balanced" if balanced else None
    model = (RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1,
                                    class_weight=weights, random_state=seed)
             if classifier == "rf" else LogisticRegression(max_iter=2000,
                                                           class_weight=weights))
    print(f"  addestramento del classificatore ({classifier}, "
          f"{'bilanciato, ' if balanced else ''}5 fold)...", flush=True)
    y_pred = cross_val_predict(model, feats, y, cv=5)
    print(f"  accuratezza {np.mean(y_pred == y):.3f}, "
          f"FPR {np.mean(y_pred[y == 0] == 1):.3f}, positivi reali {y.mean():.3f}")

    disc["fp"] = np.where(y == 0, (y_pred == 1).astype(float), np.nan)
    disc.to_pickle(cache)
    return disc, attrs, name


def diabetes(n_attrs=None):
    df, attrs, name = load_diabetes(n_attrs=n_attrs)
    return df, attrs, "fp", "bool", name.split("_rf")[0]
