"""Census-Income (KDD): censimento USA 1994-1995 (Current Population Survey).

Dati: UCI Machine Learning Repository, dataset 117,
https://archive.ics.uci.edu/dataset/117/census+income+kdd
Scaricare lo zip, estrarre (anche l'archivio census.tar.gz che contiene) e mettere in
datasets/census_kdd/ i file
    census-income.data   (199.523 righe)   oppure census-income.data.gz
    census-income.test   ( 99.762 righe)   oppure census-income.test.gz
I file non hanno intestazione: 42 colonne separate da virgola (40 attributi, il peso
statistico e la classe "- 50000." / "50000+.").

Preparazione (stessa logica di datasets_acs.py):
- si uniscono training e test (299.285 persone) e si tengono gli adulti (eta' >= 18);
- target: reddito > 50.000 $; classificatore random forest in cross-validation a
  5 fold, quindi ogni previsione e' su dati non visti. I redditi alti sono pochi
  (circa 8,6%): il classificatore e' BILANCIATO (class_weight="balanced"), altrimenti
  non farebbe quasi mai falsi positivi (FPR circa 1%) e non ci sarebbero sottogruppi
  con FPR divergente;
- outcome "fp": 1 = falso positivo, 0 = vero negativo, NaN = classe reale positiva;
- si tolgono: il peso statistico (la documentazione dice di non usarlo per la
  classificazione), l'anno (94/95) e i codici dettagliati di industria e occupazione
  (52 e 47 valori numerici: restano le versioni "major", con nomi);
- attributi numerici discretizzati a mano: eta', salario orario, guadagni e perdite
  in conto capitale, dividendi, settimane lavorate;
- valori presenti in meno dell'1% delle persone -> "other" (stati e paesi di
  nascita, ad esempio), valori mancanti ("?") -> "NA";
- attributi: sempre eta', istruzione e sesso, poi fino a n_attrs gli altri in ordine
  di entropia decrescente (come per ACS): con tutti si arriva a 37.
Il dataset preparato viene salvato in benchmarks/data_cache/ e riusato.
Requisiti: scikit-learn.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "datasets", "census_kdd")
CACHE_DIR = os.path.join(HERE, "data_cache")

COLUMNS = [
    "age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode",
    "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_stat",
    "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex",
    "member_of_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
    "capital_gains", "capital_losses", "dividends_from_stocks", "tax_filer_stat",
    "region_of_previous_residence", "state_of_previous_residence",
    "detailed_household_and_family_stat", "detailed_household_summary_in_household",
    "instance_weight", "migration_code_change_in_msa", "migration_code_change_in_reg",
    "migration_code_move_within_reg", "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
    "family_members_under_18", "country_of_birth_father", "country_of_birth_mother",
    "country_of_birth_self", "citizenship", "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin", "veterans_benefits", "weeks_worked_in_year",
    "year", "income"]
DROP = ["instance_weight", "year", "detailed_industry_recode", "detailed_occupation_recode"]
FIXED = ["age", "education", "sex"]


def _read(name):
    for suffix in ("", ".gz"):
        path = os.path.join(DATA_DIR, name + suffix)
        if os.path.exists(path):
            return pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True,
                               na_values=["?"], compression="infer")
    raise SystemExit(f"Manca {os.path.join(DATA_DIR, name)}: vedi le istruzioni in cima a "
                     "datasets_census_kdd.py")


def _cut(s, edges, labels):
    return pd.cut(s, edges, labels=labels).astype(str)


def _zero_or_terciles(s):
    """Per colonne quasi sempre 0 (guadagni, dividendi, salario): "0" oppure, per i
    valori > 0, tre fasce con lo stesso numero di persone."""
    out = pd.Series("0", index=s.index, dtype=object)
    pos = s > 0
    if pos.sum() >= 3:
        q = np.unique(np.quantile(s[pos], [1 / 3, 2 / 3]))
        edges = [0] + list(q) + [np.inf]
        labels = [f"({edges[i]:g}-{edges[i + 1]:g}]" if np.isfinite(edges[i + 1])
                  else f">{edges[i]:g}" for i in range(len(edges) - 1)]
        out[pos] = pd.cut(s[pos], edges, labels=labels).astype(str)
    return out


def _prepare(raw):
    df = raw.drop(columns=DROP)
    df = df[df["age"] >= 18].reset_index(drop=True)
    y = df["income"].astype(str).str.contains(r"50000\+").astype(int).to_numpy()
    num = df[["age", "wage_per_hour", "capital_gains", "capital_losses",
              "dividends_from_stocks", "weeks_worked_in_year"]].copy()

    disc = pd.DataFrame(index=df.index)
    disc["age"] = _cut(df["age"], [17, 24, 34, 44, 54, 64, 200],
                       ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    disc["weeks_worked_in_year"] = _cut(df["weeks_worked_in_year"], [-1, 0, 26, 51, 52],
                                        ["0", "1-26", "27-51", "52"])
    for c in ("wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks"):
        disc[c] = _zero_or_terciles(df[c])
    for c in df.columns:
        if c in disc.columns or c == "income":
            continue
        s = df[c].astype(str).where(df[c].notna(), "NA")
        freq = s.value_counts(normalize=True)
        s = s.where(~s.isin(freq[freq < 0.01].index), "other")
        disc[c] = s
    return disc, y, num


def load_census_kdd(n_attrs=37, classifier="rf", balanced=True, seed=0):
    """Restituisce (df, attributi, nome). df ha gli attributi (stringhe) e la colonna fp."""
    name = f"census_kdd_A{n_attrs}_{classifier}{'_bal' if balanced else ''}_seed{seed}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        print(f"Census KDD dalla cache: {cache}")
        return df, [c for c in df.columns if c != "fp"], name

    raw = pd.concat([_read("census-income.data"), _read("census-income.test")],
                    ignore_index=True)
    disc, y, num = _prepare(raw)

    # attributi: fissi + gli altri in ordine di entropia
    def entropy(s):
        p = s.value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())
    others = sorted((c for c in disc.columns if c not in FIXED),
                    key=lambda c: entropy(disc[c]), reverse=True)
    attrs = FIXED + others[:max(0, n_attrs - len(FIXED))]
    disc = disc[attrs].copy()
    print(f"  {len(disc)} adulti, {len(attrs)} attributi: {attrs}")

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


def census_kdd(n_attrs):
    df, attrs, _ = load_census_kdd(n_attrs=n_attrs)
    return df, attrs, "fp", "bool", f"census_kdd_A{n_attrs}"
