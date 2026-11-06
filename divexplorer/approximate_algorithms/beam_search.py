"""Algoritmo 2: beam search modificata (riserva + diversita').

  1. livello 1: tutti gli item frequenti (esaustivo);
  2. punteggio di un sottogruppo = |divergenza| se e' significativo (t >= tau),
     altrimenti 0;
  3. si scelgono i w con punteggio piu' alto che siano DIVERSI tra loro
     (sovrapposizione di righe, Jaccard <= theta); gli altri vanno nella riserva
     (coda ordinata per punteggio);
  4. ogni sottogruppo scelto si estende di una condizione (attributi non usati);
  5. se un livello resta vuoto, si riparte dalla riserva con la STESSA regola:
     si scorre in ordine di punteggio e si prende un sottogruppo solo se e' diverso
     da quelli gia' presi (i saltati restano nella riserva);
  6. stop: riserva vuota, pazienza (K passi di fila senza sottogruppi importanti
     nuovi), oppure budget di tempo / valutazioni.
Un "passo" e' un'estensione del beam (punto 4) o una ripartenza dalla riserva.
"""
import heapq
import itertools
import time

import numpy as np

from .common import Evaluator, is_important, to_dataframe


def jaccard(a, b):
    inter = np.count_nonzero(a & b)
    return inter / (np.count_nonzero(a) + np.count_nonzero(b) - inter)


def beam_search(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05,
                beam_width=10, theta=0.5, K=200, max_len=None, time_budget=None,
                max_evaluations=None):
    t0 = time.perf_counter()
    ev = Evaluator(df, attributes, outcome, kind)
    max_len = max_len or len(attributes)
    found, found_at, seen = {}, {}, set()
    reserve, tie = [], itertools.count()
    new_important = False                    # trovato un importante nel passo corrente?

    def score(itemset):
        m = ev.measures(found[itemset])
        return abs(m["div"]) if m["t"] >= tau else 0.0

    def over_budget():
        return ((time_budget is not None and time.perf_counter() - t0 > time_budget)
                or (max_evaluations is not None and ev.evaluations >= max_evaluations))

    def add(itemset):
        """Valuta un sottogruppo nuovo; True se e' frequente."""
        nonlocal new_important
        if itemset in seen:
            return False
        seen.add(itemset)
        s = ev.stats(itemset)
        if s[0] / ev.n < min_support:
            return False
        found[itemset] = s
        found_at[itemset] = time.perf_counter() - t0
        if is_important(ev, itemset, tau, eps):
            new_important = True
        return True

    def diverse(ranked):
        """Scorre i candidati in ordine di punteggio e prende al massimo w sottogruppi
        diversi tra loro. Restituisce (presi, scartati)."""
        beam, masks, rest = [], [], []
        for c in ranked:
            if len(beam) < beam_width:
                mk = ev.mask(c)
                if all(jaccard(mk, other) <= theta for other in masks):
                    beam.append(c)
                    masks.append(mk)
                    continue
            rest.append(c)
        return beam, rest

    def to_reserve(itemsets):
        for c in itemsets:
            heapq.heappush(reserve, (-score(c), next(tie), c))

    def from_reserve():
        """Ripartenza: stessa regola di diversita' applicata alla riserva, in ordine
        di punteggio. Ci si ferma quando il beam e' pieno; i saltati tornano nella
        riserva."""
        beam, masks, skipped = [], [], []
        # si guardano al massimo 50 * w sottogruppi, per non scorrere tutta la riserva
        while reserve and len(beam) < beam_width and len(skipped) < 50 * beam_width:
            entry = heapq.heappop(reserve)
            mk = ev.mask(entry[2])
            if all(jaccard(mk, other) <= theta for other in masks):
                beam.append(entry[2])
                masks.append(mk)
            else:
                skipped.append(entry)
        for entry in skipped:
            heapq.heappush(reserve, entry)
        return beam

    # 1. livello 1
    level = [frozenset([i]) for i in range(len(ev.items)) if add(frozenset([i]))]
    beam, rest = diverse(sorted(level, key=score, reverse=True))
    to_reserve(rest)
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
        # 4. estensione di una condizione
        candidates = []
        for I in beam:
            if len(I) >= max_len:
                continue
            used = {ev.item_attr[i] for i in I}
            for i in range(len(ev.items)):
                if ev.item_attr[i] not in used and add(I | {i}):
                    candidates.append(I | {i})
            if over_budget():
                break
        if candidates:
            beam, rest = diverse(sorted(candidates, key=score, reverse=True))
            to_reserve(rest)
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
