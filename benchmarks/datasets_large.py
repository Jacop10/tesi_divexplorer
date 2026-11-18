"""Dataset grandi: qui DivExplorer diventa lento, usa molta memoria o va in crash.

- synthetic25: lo stesso generatore di datasets_small, 25 attributi, 50.000 righe;
- acs25, acs30, acs35: censimento USA (California 2018, 100.000 adulti), outcome FPR
  di un random forest (vedi datasets_acs.py);
- airbnb: annunci Airbnb di New York (24 attributi, voto delle recensioni), da usare
  con supporto basso (0.05, 0.02, 0.01), vedi datasets_airbnb.py;
- kdd20, kdd25, kdd30, kdd37: Census-Income KDD (censimento USA 1994-95, circa
  200.000 adulti), FPR di un random forest su reddito > 50.000 $, vedi
  datasets_census_kdd.py;
- acs15, acs20: gli stessi di datasets_medium, per le prove con supporto minimo
  piu' basso (--min-support 0.02 o 0.01), dove anche con pochi attributi i
  sottogruppi frequenti esplodono.

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
from datasets_airbnb import airbnb
from datasets_census_kdd import census_kdd
from datasets_medium import acs, synthetic_medium

LOADERS = {
    "synthetic25": lambda: synthetic_medium(25),
    "acs15": lambda: acs(15),
    "acs20": lambda: acs(20),
    "acs25": lambda: acs(25),
    "acs30": lambda: acs(30),
    "acs35": lambda: acs(35),
    "airbnb": airbnb,
    "kdd20": lambda: census_kdd(20),
    "kdd25": lambda: census_kdd(25),
    "kdd30": lambda: census_kdd(30),
    "kdd37": lambda: census_kdd(37),
}
