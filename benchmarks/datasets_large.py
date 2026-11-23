"""Dataset grandi: qui DivExplorer diventa lento, usa molta memoria o va in crash.

- synthetic25: lo stesso generatore di datasets_small, 25 attributi, 50.000 righe;
- acs25, acs30, acs35: censimento USA (California 2018, 100.000 adulti), outcome FPR
  di un random forest (vedi datasets_acs.py);
- airbnb: annunci Airbnb di New York (24 attributi, voto delle recensioni), da usare
  con supporto basso (0.05, 0.02, 0.01), vedi datasets_airbnb.py;
- airbnb2_20, airbnb2_25, airbnb2_30, airbnb2_all: Airbnb con il nostro
  preprocessing (17.409 annunci con almeno 5 recensioni, fino a 74 attributi), vedi
  preprocess_airbnb.py;
- kdd20, kdd25, kdd30, kdd37: Census-Income KDD (censimento USA 1994-95, circa
  200.000 adulti), FPR di un random forest su reddito > 50.000 $, vedi
  datasets_census_kdd.py;
- diabetes20, diabetes25, diabetes30, diabetes_all: Diabetes 130-US hospitals (circa
  100.000 ricoveri), FPR di un random forest sulla riammissione entro 30 giorni, vedi
  datasets_diabetes.py (diabetes_all = tutti gli attributi non quasi costanti);
- hmda20, hmda25, hmda30, hmda35, hmda40, hmda_all: richieste di mutuo (HMDA, New York 2022); outcome
  = tasso di RIFIUTO reale della banca (non l'errore di un classificatore), vedi
  datasets_hmda.py;
- acs15, acs20: gli stessi di datasets_medium, per le prove con supporto minimo
  piu' basso (--min-support 0.02 o 0.01), dove anche con pochi attributi i
  sottogruppi frequenti esplodono.

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
from datasets_airbnb import airbnb, airbnb_v2
from datasets_census_kdd import census_kdd
from datasets_diabetes import diabetes
from datasets_hmda import hmda
from datasets_medium import acs, synthetic_medium

LOADERS = {
    "synthetic25": lambda: synthetic_medium(25),
    "acs15": lambda: acs(15),
    "acs20": lambda: acs(20),
    "acs25": lambda: acs(25),
    "acs30": lambda: acs(30),
    "acs35": lambda: acs(35),
    "airbnb": airbnb,
    "airbnb2_20": lambda: airbnb_v2(20),
    "airbnb2_25": lambda: airbnb_v2(25),
    "airbnb2_30": lambda: airbnb_v2(30),
    "airbnb2_all": lambda: airbnb_v2(None),
    "kdd20": lambda: census_kdd(20),
    "kdd25": lambda: census_kdd(25),
    "kdd30": lambda: census_kdd(30),
    "kdd37": lambda: census_kdd(37),
    "diabetes20": lambda: diabetes(20),
    "diabetes25": lambda: diabetes(25),
    "diabetes30": lambda: diabetes(30),
    "diabetes_all": lambda: diabetes(None),
    "hmda20": lambda: hmda(20),
    "hmda25": lambda: hmda(25),
    "hmda30": lambda: hmda(30),
    "hmda35": lambda: hmda(35),
    "hmda40": lambda: hmda(40),
    "hmda_all": lambda: hmda(None),
}
