"""Algoritmo 1: esplorazione random delle righe (matrice compatta + pazienza K).

Ripeti:
  1. scegli la direzione (+ divergenza positiva, - negativa) a caso;
  2. scegli una riga r con probabilita' proporzionale al suo peso
     (booleano: 1 se outcome = 1 per +, se outcome = 0 per -;
      numerico: y - min per +, max - y per -);
  3. scegli m attributi a caso (nuovi a ogni campione);
  4. itemset campionato I_r = valori della riga r su quegli m attributi;
  5. matrice compatta: per ogni riga del dataset il codice c (bit j = 1 se la riga
     ha lo stesso valore di r sull'attributo j), istogramma H[c] = (righe, w1, w2),
     poi S = Z^T H con Z[c, J] = 1 se J e' contenuto in c. S[J] sono le statistiche
     di TUTTI i 2^m - 1 sottogruppi di I_r;
  6. salva i sottogruppi frequenti nuovi; se tra questi c'e' un sottogruppo
     importante (t >= tau e non ridondante) azzera il contatore.
Si ferma dopo K campioni di fila senza nuovi sottogruppi importanti.
"""
import time

import numpy as np

from .common import Evaluator, important_itemsets, measures, to_dataframe

_Z = {}


def zeta_matrix(m):
    """Z[c, J] = 1 se il sottogruppo J (in bit) e' contenuto nella combinazione c."""
    if m not in _Z:
        idx = np.arange(2 ** m, dtype=np.int64)
        _Z[m] = ((idx[:, None] & idx[None, :]) == idx[None, :]).astype(np.float64)
    return _Z[m]


def random_rows(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05,
                m=10, K=2000, max_samples=None, time_budget=None, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    ev = Evaluator(df, attributes, outcome, kind)
    n, A = ev.n, len(attributes)
    m = min(m, A)
    Z = zeta_matrix(m)
    codes_all = np.arange(2 ** m, dtype=np.int64)

    # item di ogni riga per ogni attributo (-1 se il valore manca)
    row_items = -np.ones((n, A), dtype=np.int64)
    for i, (a, mk) in enumerate(zip(ev.item_attr, ev.masks)):
        row_items[mk, a] = i

    # pesi per scegliere le righe nelle due direzioni
    y = ev.y
    if kind == "bool":
        weights = {+1: (y == 1).astype(float), -1: (y == 0).astype(float)}
    else:
        weights = {+1: y - y.min(), -1: y.max() - y}
    cum = {d: np.cumsum(w) for d, w in weights.items()}

    found = {}                 # frozenset di item -> (righe, w1, w2)
    important = set()
    samples, since_new = 0, 0
    stopped_by = "pazienza"
    while since_new < K:
        if max_samples is not None and samples >= max_samples:
            stopped_by = "max_samples"
            break
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            stopped_by = "tempo"
            break
        samples += 1
        since_new += 1

        # 1-4. direzione, riga, attributi -> itemset campionato
        d = rng.choice([+1, -1])
        r = int(np.searchsorted(cum[d], rng.random() * cum[d][-1], side="right"))
        avail = np.nonzero(row_items[r] >= 0)[0]
        chosen = rng.choice(avail, size=min(m, len(avail)), replace=False)
        items = row_items[r, chosen]
        mm = len(items)

        # 5. matrice compatta
        code = np.zeros(n, dtype=np.int64)
        for j, it in enumerate(items):
            code[ev.masks[it]] |= 1 << j
        H = np.stack([np.bincount(code, weights=ev.W[:, k], minlength=2 ** m)
                      for k in range(3)], axis=1)
        S = Z.T @ H                                   # statistiche dei 2^m sottogruppi
        if mm < m:                                    # riga con valori mancanti
            S = S[: 2 ** mm]
        codes = codes_all[: 2 ** mm]

        # 6. sottogruppi frequenti nuovi e importanti
        meas = measures(S, ev.root, n, kind)
        div = np.where(codes == 0, 0.0, meas["div"])
        freq = (meas["support"] >= min_support) & (codes > 0)
        nonred = np.ones(len(codes), dtype=bool)
        for j in range(mm):
            has = (codes >> j) & 1 == 1
            nonred &= ~has | (np.abs(div - div[codes ^ (1 << j)]) > eps)
        imp = freq & (meas["t"] >= tau) & nonred

        for c in np.nonzero(freq)[0]:
            itemset = frozenset(int(items[j]) for j in range(mm) if (c >> j) & 1)
            if itemset not in found:
                found[itemset] = S[c]
                if imp[c]:
                    important.add(itemset)
                    since_new = 0

    res = to_dataframe(ev, found)
    info = {"tempo": time.perf_counter() - t0, "campioni": samples,
            "frequenti": len(found), "fermato_da": stopped_by}
    return res, ev, info


def important_of(res, ev, tau, eps):
    """Con la chiusura verso il basso tutti i sottoinsiemi sono gia' in res."""
    return important_itemsets(res, ev.outcome, tau, eps, ev)
