"""Algoritmo 2: beam search modificata (riserva + diversita').

  1. livelli 1 e 2 ESAUSTIVI: tutti gli item frequenti e tutte le coppie frequenti
     (parametro exhaustive_levels, default 2);
  2. punteggio di un sottogruppo = |divergenza| se e' significativo (t >= tau),
     altrimenti 0;
  3. dall'ultimo livello esaustivo in poi si scelgono w sottogruppi con punteggio alto
     e DIVERSI tra loro, con la selezione "cover-based" di DSSD (van Leeuwen &
     Knobbe, 2012): si sceglie ogni volta il sottogruppo con punteggio x peso piu'
     alto, dove il peso e' la media sulle sue righe di alpha^c (c = quanti sottogruppi
     gia' scelti coprono quella riga, alpha = 0.9). Le righe gia' coperte penalizzano
     un candidato, ma non lo escludono. Gli altri vanno nella riserva (coda ordinata
     per punteggio);
  4. ogni sottogruppo scelto si estende di una condizione (attributi non usati);
  5. se un livello resta vuoto, si riparte dalla riserva con la STESSA regola di
     diversita' (i saltati restano nella riserva);
  6. stop: riserva vuota, pazienza (K passi di fila senza sottogruppi importanti
     nuovi), oppure budget di tempo / valutazioni.
Un "passo" e' un'estensione del beam (punto 4) o una ripartenza dalla riserva.

Valutazione veloce: tutte le estensioni di un sottogruppo si calcolano insieme, con
un solo passaggio sulle sue righe (Evaluator.extensions).
"""
import heapq
import itertools
import time

import numpy as np
import scipy.sparse as sp

from .common import Evaluator, is_important, to_dataframe


def beam_search(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05,
                beam_width=10, alpha=0.9, K=200, max_len=None, time_budget=None,
                max_evaluations=None, exhaustive_levels=2):
    t0 = time.perf_counter()
    ev = Evaluator(df, attributes, outcome, kind)
    max_len = max_len or len(attributes)
    found, found_at, seen = {}, {}, set()
    reserve, tie = [], itertools.count()
    new_important = False                    # trovato un importante nel passo corrente?
    pool = 20 * beam_width                   # candidati considerati per la scelta

    def score(itemset):
        m = ev.measures(found[itemset])
        return abs(m["div"]) if m["t"] >= tau else 0.0

    def over_budget():
        return ((time_budget is not None and time.perf_counter() - t0 > time_budget)
                or (max_evaluations is not None and ev.evaluations >= max_evaluations))

    def extend(I):
        """Tutte le estensioni di I in un passaggio; restituisce quelle frequenti nuove."""
        nonlocal new_important
        if len(I) >= max_len:
            return []
        out = []
        for i, s in ev.extensions(I).items():
            J = I | {i}
            if J in seen:
                continue
            seen.add(J)
            if s[0] / ev.n < min_support:
                continue
            found[J] = s
            found_at[J] = time.perf_counter() - t0
            if is_important(ev, J, tau, eps):
                new_important = True
            out.append(J)
        return out

    # --- scelta di sottogruppi diversi -------------------------------------------
    def pick(ranked):
        """Selezione cover-based di DSSD sui primi `pool` candidati per punteggio.
        ranked: candidati in ordine di punteggio. Restituisce (presi, non presi).
        I pesi di tutti i candidati si aggiornano insieme con un prodotto
        matrice-vettore (matrice candidati x righe, sparsa)."""
        head, tail = ranked[:pool], ranked[pool:]
        if len(head) <= beam_width:
            return head, tail
        rows = [ev.rows(c) for c in head]
        sizes = np.array([max(len(r), 1) for r in rows], dtype=float)
        M = sp.csr_matrix((np.ones(sum(len(r) for r in rows)),
                           np.concatenate(rows),
                           np.concatenate([[0], np.cumsum([len(r) for r in rows])])),
                          shape=(len(head), ev.n))
        sc = np.array([score(c) for c in head])
        counts = np.zeros(ev.n)
        free = np.ones(len(head), dtype=bool)
        chosen = []
        for _ in range(beam_width):
            weight = (M @ (alpha ** counts)) / sizes        # media di alpha^c sulle righe
            val = np.where(free, sc * weight, -1.0)
            j = int(np.argmax(val))
            chosen.append(j)
            free[j] = False
            counts[rows[j]] += 1
        beam = [head[j] for j in chosen]
        rest = [head[j] for j in range(len(head)) if free[j]] + tail
        return beam, rest

    def select(candidates):
        beam, rest = pick(sorted(candidates, key=score, reverse=True))
        for c in rest:
            heapq.heappush(reserve, (-score(c), next(tie), c))
        return beam

    def from_reserve():
        """Ripartenza dalla riserva con la stessa regola di diversita': si prendono i
        primi della riserva, si sceglie, i non scelti tornano nella riserva."""
        top = [heapq.heappop(reserve)[2] for _ in range(min(len(reserve), 50 * beam_width))]
        return select(top)

    # 1. livelli esaustivi
    level = extend(frozenset())                              # livello 1
    for _ in range(exhaustive_levels - 1):                   # livello 2, ...
        nxt = []
        for I in level:
            nxt += extend(I)
        if not nxt:
            break
        level = nxt
    beam = select(level)
    steps, since_new = 0, 0
    stopped_by = "riserva vuota"
    while True:
        if over_budget():
            stopped_by = "budget"
            break
        if since_new >= K:
            stopped_by = "pazienza"
            break
        steps += 1
        new_important = False
        # 4. estensione di una condizione (tutte le estensioni insieme)
        candidates = []
        for I in beam:
            candidates += extend(I)
            if over_budget():
                break
        if candidates:
            beam = select(candidates)
        else:
            # 5. livello vuoto: si riparte dalla riserva
            if not reserve:
                break
            beam = from_reserve()
        since_new = 0 if new_important else since_new + 1

    res = to_dataframe(ev, found, found_at)
    info = {"tempo": time.perf_counter() - t0, "passi": steps,
            "valutazioni": ev.evaluations, "frequenti": len(found),
            "riserva": len(reserve), "fermato_da": stopped_by}
    return res, ev, info
