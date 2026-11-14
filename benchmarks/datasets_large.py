"""Dataset grandi: qui DivExplorer diventa lento, usa molta memoria o va in crash.

- synthetic25: lo stesso generatore di datasets_small, 25 attributi, 50.000 righe;
- acs25, acs30, acs35: censimento USA (California 2018, 100.000 adulti), outcome FPR
  di un random forest (vedi datasets_acs.py);
- acs15, acs20: gli stessi di datasets_medium, per le prove con supporto minimo
  piu' basso (--min-support 0.02 o 0.01), dove anche con pochi attributi i
  sottogruppi frequenti esplodono.

Ogni loader restituisce (df, attributi, outcome, tipo, nome).
"""
from datasets_medium import acs, synthetic_medium

LOADERS = {
    "synthetic25": lambda: synthetic_medium(25),
    "acs15": lambda: acs(15),
    "acs20": lambda: acs(20),
    "acs25": lambda: acs(25),
    "acs30": lambda: acs(30),
    "acs35": lambda: acs(35),
}
