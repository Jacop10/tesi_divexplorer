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

Due scelte per tenere bassi memoria e tempo (come in random_explorer.py):
- ogni sottogruppo frequente e' salvato come un intero (un bit per item) e le sue
  statistiche in tre array di double: circa 100 byte per sottogruppo;
- la chiave di un sottogruppo si ottiene da quella del sottogruppo con un item in
  meno con un solo OR, senza ricostruire l'insieme da zero.

evaluation:
- "matrix" (default, baseline): S = Z^T H, costo O(4^m) per campione;
- "zeta" (ottimizzazione): somma sui superinsiemi un item alla volta, O(m 2^m).
I risultati sono identici.

output:
- "all" (default): si salvano TUTTI i sottogruppi frequenti trovati, come
  DivExplorer. Serve per lo Shapley globale e per usare DivergencePatternProcessor
  sul risultato. La memoria cresce con il numero di frequenti (milioni sui dataset
  grandi);
- "important": si salvano SOLO i sottogruppi importanti. Basta, perche' un
  sottogruppo si riconosce come importante dentro il campione (ci sono tutti i suoi
  sottoinsiemi) e la pazienza conta solo gli importanti nuovi. Risultati identici
  sugli importanti, memoria molto piu' bassa; non si ha la lista dei frequenti.
"""
import time
from array import array

import numpy as np
import pandas as pd

from .common import Evaluator, measures

_Z = {}


def zeta_matrix(m):
    """Z[c, J] = 1 se il sottogruppo J (in bit) e' contenuto nella combinazione c."""
    if m not in _Z:
        idx = np.arange(2 ** m, dtype=np.int64)
        _Z[m] = ((idx[:, None] & idx[None, :]) == idx[None, :]).astype(np.float64)
    return _Z[m]


def superset_sum(H, m):
    """Trucco (fast zeta transform): G[J] = somma di H[c] per ogni c che contiene J.
    Stesso risultato di zeta_matrix(m).T @ H."""
    G = H.copy()
    for j in range(m):
        G = G.reshape(-1, 2, 2 ** j, H.shape[-1])
        G[:, 0, :, :] += G[:, 1, :, :]
        G = G.reshape(-1, H.shape[-1])
    return G


def random_rows(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05,
                m=10, K=2000, max_samples=None, time_budget=None,
                evaluation="matrix", output="all", seed=0, verbose=False):
    assert evaluation in ("matrix", "zeta")
    assert output in ("all", "important")
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    ev = Evaluator(df, attributes, outcome, kind)
    n, A = ev.n, len(attributes)
    m = min(m, A)

    # item di ogni riga per ogni attributo (-1 se il valore manca)
    row_items = -np.ones((n, A), dtype=np.int64)
    for i, (a, mk) in enumerate(zip(ev.item_attr, ev.masks)):
        row_items[mk, a] = i
    item_rows = [np.nonzero(mk)[0] for mk in ev.masks]     # righe di ogni item

    # pesi per scegliere le righe nelle due direzioni
    y = ev.y
    if kind == "bool":
        weights = {+1: (y == 1).astype(float), -1: (y == 0).astype(float)}
    else:
        weights = {+1: y - y.min(), -1: y.max() - y}
    samplers = {}
    for d, w in weights.items():
        rows = np.nonzero(w > 0)[0]
        cdf = np.cumsum(w[rows])
        samplers[d] = (rows, cdf / cdf[-1])

    # memoria compatta: chiave intera -> posizione negli array delle statistiche
    # (output="all": tutti i frequenti; output="important": solo gli importanti)
    index = {}
    cnt_a, w1_a, w2_a = array("d"), array("d"), array("d")
    t_a = array("d")                                   # quando e' stato trovato
    samples, since_new, n_important = 0, 0, 0
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
        d = +1 if rng.random() < 0.5 else -1
        rows, cdf = samplers[d]
        r = rows[min(int(np.searchsorted(cdf, rng.random(), side="right")), len(rows) - 1)]
        avail = np.nonzero(row_items[r] >= 0)[0]
        mm = min(m, len(avail))
        if mm == 0:
            continue
        items = [int(i) for i in row_items[r, rng.choice(avail, size=mm, replace=False)]]

        # 5. matrice compatta: codici, istogramma, statistiche di tutti i sottogruppi
        code = np.zeros(n, dtype=np.int64)
        for j, it in enumerate(items):
            code[item_rows[it]] |= 1 << j
        H = np.stack([np.bincount(code, weights=ev.W[:, k], minlength=2 ** mm)
                      for k in range(3)], axis=1)
        S = zeta_matrix(mm).T @ H if evaluation == "matrix" else superset_sum(H, mm)

        # 6. frequenti e importanti (t >= tau e non ridondanti)
        codes = np.arange(2 ** mm, dtype=np.int64)
        meas = measures(S, ev.root, n, kind)
        div = np.where(codes == 0, 0.0, np.nan_to_num(meas["div"]))
        freq = (meas["support"] >= min_support) & (codes > 0)
        nonred = np.ones(len(codes), dtype=bool)
        for j in range(mm):
            has = (codes >> j) & 1 == 1
            nonred &= ~has | (np.abs(div - div[codes ^ (1 << j)]) > eps)
        imp = freq & (np.nan_to_num(meas["t"]) >= tau) & nonred

        bits = [1 << it for it in items]
        if output == "all":
            # chiavi: si scorrono i codici in ordine crescente, cosi' c senza il suo
            # bit piu' basso (un sottoinsieme, quindi anch'esso frequente) e' gia'
            # calcolato
            keys = [0] * (2 ** mm)
            todo = []
            for c in np.nonzero(freq)[0].tolist():
                low = c & -c
                keys[c] = keys[c ^ low] | bits[low.bit_length() - 1]
                todo.append((c, keys[c]))
        else:
            # solo gli importanti: la chiave si costruisce bit per bit (al massimo m)
            todo = []
            for c in np.nonzero(imp)[0].tolist():
                k, cc = 0, c
                while cc:
                    low = cc & -cc
                    k |= bits[low.bit_length() - 1]
                    cc ^= low
                todo.append((c, k))
        for c, k in todo:
            if k not in index:
                index[k] = len(cnt_a)
                cnt_a.append(S[c, 0])
                w1_a.append(S[c, 1])
                w2_a.append(S[c, 2])
                t_a.append(time.perf_counter() - t0)
                if imp[c]:
                    n_important += 1
                    since_new = 0
        if verbose and samples % 500 == 0:
            print(f"  {samples} campioni, {len(index)} sottogruppi salvati, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    # output nel formato di DivExplorer
    def decode(k):
        names = []
        while k:
            low = k & -k
            names.append(ev.items[low.bit_length() - 1])
            k ^= low
        return frozenset(names)

    itemsets = [decode(k) for k in index]              # stesso ordine degli array
    n_frequent = len(index)
    del index
    S_all = np.stack([np.frombuffer(a, dtype=float) for a in (cnt_a, w1_a, w2_a)], axis=1)
    meas = measures(S_all, ev.root, n, kind) if n_frequent else \
        {"support": [], "value": [], "div": [], "t": []}
    res = pd.DataFrame({"support": meas["support"], "itemset": itemsets,
                        outcome: meas["value"], f"{outcome}_div": meas["div"],
                        f"{outcome}_t": meas["t"],
                        "length": [len(i) for i in itemsets],
                        "t_found": np.frombuffer(t_a, dtype=float)})
    info = {"tempo": time.perf_counter() - t0, "campioni": samples,
            "frequenti": n_frequent if output == "all" else None,
            "importanti_trovati": n_important,
            "fermato_da": stopped_by, "evaluation": evaluation, "output": output}
    return res, ev, info
