"""Parti comuni a tutti gli algoritmi.

- Evaluator: calcola supporto, divergenza e t-value di QUALSIASI sottogruppo,
  con le stesse formule di DivExplorer. Lo usano beam search e MCTS, che
  valutano un sottogruppo alla volta.
- measures: da (righe, w1, w2) a supporto, outcome, divergenza e t-value.
- to_dataframe: risultato nel formato di DivExplorer.
- important_itemsets: sottogruppi significativi (t >= tau) e non ridondanti
  (redundancy pruning di DivExplorer, soglia eps);
- is_important: la stessa regola per un solo sottogruppo, durante la ricerca.

Notazione per le statistiche di un sottogruppo, s = (righe, w1, w2):
- outcome booleano: w1 = righe con outcome 1, w2 = righe con outcome 0;
- outcome numerico: w1 = somma dei valori, w2 = somma dei quadrati.
"""
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, df, attributes, outcome, kind):
        """kind = "bool" (outcome 1 / 0 / NaN) oppure "num" (outcome numerico)."""
        df = df.reset_index(drop=True)
        self.n = len(df)
        self.kind = kind
        self.outcome = outcome
        self.attributes = list(attributes)

        # item = "attributo=valore", come nel one-hot encoding di DivExplorer
        self.items, self.item_attr, self.masks = [], [], []
        for ai, a in enumerate(self.attributes):
            codes, values = pd.factorize(df[a], sort=True)        # NaN -> nessun item
            for k, v in enumerate(values):
                self.items.append(f"{a}={v}")
                self.item_attr.append(ai)
                self.masks.append(codes == k)
        self.item_id = {name: i for i, name in enumerate(self.items)}

        y = df[outcome].to_numpy(dtype=float)
        if kind == "bool":
            w1, w2 = (y == 1).astype(float), (y == 0).astype(float)
        else:
            if np.isnan(y).any():
                raise ValueError("L'outcome numerico non deve avere valori mancanti.")
            w1, w2 = y, y ** 2
        self.y = y
        self.W = np.stack([np.ones(self.n), w1, w2], axis=1)   # una riga per persona
        self.root = self.W.sum(axis=0)                          # statistiche del dataset
        self.cache = {}
        self.evaluations = 0

    # --- un sottogruppo alla volta (beam search, MCTS) ---
    def mask(self, itemset):
        m = np.ones(self.n, dtype=bool)
        for i in itemset:
            m &= self.masks[i]
        return m

    def stats(self, itemset):
        """itemset = frozenset di indici di item. Restituisce (righe, w1, w2)."""
        s = self.cache.get(itemset)
        if s is None:
            s = self.W[self.mask(itemset)].sum(axis=0)
            self.cache[itemset] = s
            self.evaluations += 1
        return s

    def measures(self, s):
        return measures(s, self.root, self.n, self.kind)

    def names(self, itemset):
        return frozenset(self.items[i] for i in itemset)


def measures(s, root, n, kind):
    """Supporto, outcome, divergenza e t-value con le formule di DivExplorer.
    Funziona sia per un sottogruppo (s di 3 numeri) sia per tanti insieme
    (s matrice k x 3): in quel caso restituisce array."""
    cnt, w1, w2 = np.asarray(s, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "bool":
            def beta(p, q):
                a, b = p + 1, q + 1
                return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))
            value = w1 / (w1 + w2)                 # NaN se nessuna riga con outcome
            value0 = root[1] / (root[1] + root[2])
            mu, var = beta(w1, w2)
            mu0, var0 = beta(root[1], root[2])
            t = np.abs(mu - mu0) / np.sqrt(var + var0)
        else:
            value, value0 = w1 / cnt, root[1] / root[0]
            var = np.maximum(w2 / cnt - value ** 2, 0)
            var0 = root[2] / root[0] - value0 ** 2
            den = np.sqrt(var / cnt + var0 / root[0])
            t = np.where(den > 0, np.abs(value - value0) / den, 0.0)
    return {"support": cnt / n, "value": value, "div": value - value0, "t": t}


def to_dataframe(ev, found, found_at=None):
    """found = {frozenset di item: (righe, w1, w2)} -> DataFrame come DivExplorer.
    found_at = {frozenset di item: secondi dall'inizio} -> colonna t_found."""
    o = ev.outcome
    rows = []
    for itemset, s in found.items():
        m = ev.measures(s)
        rows.append({"support": m["support"], "itemset": ev.names(itemset),
                     o: m["value"], f"{o}_div": m["div"], f"{o}_t": m["t"],
                     "length": len(itemset),
                     "t_found": found_at[itemset] if found_at else float("nan")})
    return pd.DataFrame(rows, columns=["support", "itemset", o, f"{o}_div", f"{o}_t",
                                       "length", "t_found"])


def is_important(ev, itemset, tau, eps):
    """Un sottogruppo (frozenset di indici) e' importante se ha t >= tau e non e'
    ridondante: togliendo qualsiasi item la divergenza cambia di piu' di eps
    (divergenza dell'insieme vuoto = 0). Come important_itemsets, ma per un solo
    sottogruppo, durante la ricerca (beam search, MCTS)."""
    m = ev.measures(ev.stats(itemset))
    if not m["t"] >= tau:
        return False
    for i in itemset:
        sub = itemset - {i}
        d_sub = ev.measures(ev.stats(sub))["div"] if sub else 0.0
        if abs(m["div"] - d_sub) <= eps:
            return False
    return True


def important_itemsets(df, outcome, tau, eps, ev=None):
    """Sottogruppi (insiemi di nomi) con t >= tau e non ridondanti: I e' ridondante se
    togliendo un item la divergenza cambia al massimo di eps (divergenza dell'insieme
    vuoto = 0). Se un sottoinsieme non e' nel DataFrame, la sua divergenza si calcola
    con l'Evaluator `ev` (serve per beam search e MCTS)."""
    div_of = dict(zip(df["itemset"], df[f"{outcome}_div"]))
    div_of[frozenset()] = 0.0

    def div(names):
        if names not in div_of:
            ids = frozenset(ev.item_id[x] for x in names)
            div_of[names] = ev.measures(ev.stats(ids))["div"]
        return div_of[names]

    out = set()
    sig = df[(df["length"] > 0) & (df[f"{outcome}_t"] >= tau)]
    for names, d in zip(sig["itemset"], sig[f"{outcome}_div"]):
        if all(abs(d - div(names - {x})) > eps for x in names):
            out.add(names)
    return out
