"""Parti comuni a tutti gli algoritmi.

- Evaluator: calcola supporto, divergenza e t-value di QUALSIASI sottogruppo,
  con le stesse formule di DivExplorer. Lo usano beam search e MCTS, che
  valutano un sottogruppo alla volta.
- measures: da (righe, w1, w2) a supporto, outcome, divergenza e t-value.
- to_dataframe: risultato nel formato di DivExplorer.
- important_itemsets: sottogruppi significativi (t >= tau) e non ridondanti
  (redundancy pruning di DivExplorer, soglia eps);
- is_important: la stessa regola per un solo sottogruppo, durante la ricerca.

Valutazione veloce (beam search e MCTS):
- Evaluator.extensions: statistiche di TUTTE le estensioni di un sottogruppo (il
  sottogruppo piu' una condizione) con un solo passaggio sulle SUE righe: per ogni
  attributo non usato si contano le righe per valore (stesso principio della
  matrice compatta di random rows). Prima ogni estensione scorreva tutte le righe
  del dataset. Con outcome booleano basta UN conteggio intero per attributo
  (chiave = item + numero di item x classe), invece di tre conteggi pesati.
- Evaluator.measures: per un solo sottogruppo usa numeri Python invece di numpy
  (stesse formule e stessi risultati, al bit, ma molto piu' veloce).
- Evaluator.rows: righe di un sottogruppo partendo dall'item piu' raro.

Notazione per le statistiche di un sottogruppo, s = (righe, w1, w2):
- outcome booleano: w1 = righe con outcome 1, w2 = righe con outcome 0;
- outcome numerico: w1 = somma dei valori, w2 = somma dei quadrati.
"""
import math

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
        # item di ogni riga per ogni attributo (-1 se il valore manca)
        self.row_items = -np.ones((self.n, len(self.attributes)), dtype=np.int64)
        for i, (a, mk) in enumerate(zip(self.item_attr, self.masks)):
            self.row_items[mk, a] = i
        # le stesse informazioni per colonna (un attributo = un array contiguo): leggere
        # la colonna di un attributo per un gruppo di righe e' molto piu' veloce
        self.cols = np.ascontiguousarray(self.row_items.T, dtype=np.int32)
        # righe di ogni item (per trovare in fretta le righe di un sottogruppo)
        self.item_rows = [np.flatnonzero(mk).astype(np.int32) for mk in self.masks]

        y = df[outcome].to_numpy(dtype=float)
        if kind == "bool":
            w1, w2 = (y == 1).astype(float), (y == 0).astype(float)
            self._cls = np.where(y == 1, 0, np.where(y == 0, 1, 2)).astype(np.int64)
        else:
            if np.isnan(y).any():
                raise ValueError("L'outcome numerico non deve avere valori mancanti.")
            w1, w2 = y, y ** 2
        self.y = y
        self.W = np.stack([np.ones(self.n), w1, w2], axis=1)   # una riga per persona
        self.root = self.W.sum(axis=0)                          # statistiche del dataset
        # costanti del dataset per il calcolo veloce delle misure di UN sottogruppo
        r = [float(v) for v in self.root]
        if kind == "bool":
            a0, b0 = r[1] + 1, r[2] + 1
            self._c = (r[1] / (r[1] + r[2]), a0 / (a0 + b0),
                       a0 * b0 / ((a0 + b0) * (a0 + b0) * (a0 + b0 + 1)))
        else:
            v0 = r[1] / r[0]
            self._c = (v0, (r[2] / r[0] - v0 ** 2) / r[0])
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
            s = self.row_stats(self.rows(itemset))
            self.cache[itemset] = s
            self.evaluations += 1
        return s

    def rows(self, itemset):
        """Indici delle righe del sottogruppo (in ordine crescente): si parte dalle
        righe dell'item piu' raro e si tengono quelle che hanno anche gli altri item."""
        if not itemset:
            return np.arange(self.n)
        items = sorted(itemset, key=lambda i: len(self.item_rows[i]))
        r = self.item_rows[items[0]]
        if len(items) > 1 and 8 * len(r) > self.n:   # item comune: piu' veloce con le maschere
            return np.flatnonzero(self.mask(itemset))
        for i in items[1:]:
            r = r[self.cols[self.item_attr[i]][r] == i]
        return r

    def row_stats(self, rows):
        """(righe, w1, w2) di un insieme di righe."""
        if self.kind == "bool":
            b = np.bincount(self._cls[rows], minlength=3)
            return np.array([len(rows), b[0], b[1]], dtype=float)
        return self.W[rows].sum(axis=0)

    def extensions(self, itemset, rows=None):
        """Statistiche di tutte le estensioni di `itemset` con un item di un attributo
        non ancora usato, calcolate con un solo passaggio sulle righe del sottogruppo.
        Restituisce {item: (righe, w1, w2)} e salva tutto nella cache."""
        if rows is None:
            rows = self.rows(itemset)
        used = {self.item_attr[i] for i in itemset}
        free = [a for a in range(len(self.attributes)) if a not in used]
        if not free or len(rows) == 0:
            return {}
        n_items = len(self.items)
        if self.kind == "bool":
            # un solo conteggio intero per attributo: chiave = item + n_items * classe
            # (classe 0: outcome 1, 1: outcome 0, 2: outcome mancante)
            B = np.zeros(3 * n_items, dtype=np.int64)
            off = self._cls[rows] * n_items
            for a in free:
                c = self.cols[a][rows]
                ok = c >= 0
                key = c + off
                B += np.bincount(key if ok.all() else key[ok], minlength=3 * n_items)
            B = B.reshape(3, n_items)
            S = np.column_stack([B.sum(axis=0), B[0], B[1]]).astype(float)
        else:
            Wr = self.W[rows]
            S = np.zeros((n_items, 3))
            for a in free:                      # un conteggio per attributo
                c = self.cols[a][rows]
                ok = c >= 0
                w = Wr
                if not ok.all():
                    c, w = c[ok], Wr[ok]
                for k in range(3):
                    S[:, k] += np.bincount(c, weights=w[:, k], minlength=n_items)
        out = {}
        for i in np.flatnonzero(S[:, 0] > 0):
            J = itemset | {int(i)}
            if J not in self.cache:
                self.cache[J] = S[i]
                self.evaluations += 1
            out[int(i)] = self.cache[J]
        return out

    def measures(self, s):
        """Come measures(), ma per UN sottogruppo il calcolo e' fatto con numeri
        Python (molto piu' veloce, stessi risultati)."""
        if np.ndim(s) != 1:
            return measures(s, self.root, self.n, self.kind)
        cnt, w1, w2 = float(s[0]), float(s[1]), float(s[2])
        if self.kind == "bool":
            value0, mu0, var0 = self._c
            value = w1 / (w1 + w2) if w1 + w2 > 0 else math.nan
            a, b = w1 + 1, w2 + 1
            mu = a / (a + b)
            var = a * b / ((a + b) * (a + b) * (a + b + 1))
            t = abs(mu - mu0) / math.sqrt(var + var0)
        else:
            value0, v0n = self._c
            value = w1 / cnt
            var = max(w2 / cnt - value ** 2, 0.0)
            den = math.sqrt(var / cnt + v0n)
            t = abs(value - value0) / den if den > 0 else 0.0
        return {"support": cnt / self.n, "value": value, "div": value - value0, "t": t}

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
