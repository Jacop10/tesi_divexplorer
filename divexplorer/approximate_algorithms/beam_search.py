"""Algoritmo 2: beam search modificata (riserva + diversita').

  1. livello 1: tutti gli item frequenti (esaustivo);
  2. si ordinano i candidati per |divergenza| e si scelgono i w migliori che siano
     DIVERSI tra loro (sovrapposizione di righe, Jaccard <= theta);
  3. gli altri candidati vanno nella riserva (coda con priorita' |divergenza|);
  4. ogni sottogruppo scelto si estende di una condizione (attributi non usati);
  5. se un livello resta vuoto, si riparte dai w migliori della riserva;
  6. stop: riserva vuota, lunghezza massima, oppure budget di tempo/valutazioni.
"""
import heapq
import itertools
import time

import numpy as np

from .common import Evaluator, to_dataframe


def jaccard(a, b):
    inter = np.count_nonzero(a & b)
    return inter / (np.count_nonzero(a) + np.count_nonzero(b) - inter)


def beam_search(df, attributes, outcome, kind, min_support, beam_width=10, theta=0.5,
                max_len=None, time_budget=None, max_evaluations=None):
    t0 = time.perf_counter()
    ev = Evaluator(df, attributes, outcome, kind)
    max_len = max_len or len(attributes)
    found, seen = {}, set()
    reserve, tie = [], itertools.count()

    def score(itemset):
        return abs(ev.measures(found[itemset])["div"])

    def over_budget():
        return ((time_budget is not None and time.perf_counter() - t0 > time_budget)
                or (max_evaluations is not None and ev.evaluations >= max_evaluations))

    def add(itemset):
        """Valuta un sottogruppo nuovo; True se e' frequente."""
        if itemset in seen:
            return False
        seen.add(itemset)
        s = ev.stats(itemset)
        if s[0] / ev.n >= min_support:
            found[itemset] = s
            return True
        return False

    def select_diverse(candidates):
        """I w migliori e diversi tra loro; tutti gli altri vanno nella riserva."""
        candidates = sorted(candidates, key=score, reverse=True)
        beam, masks = [], []
        for c in candidates:
            if len(beam) < beam_width:
                mk = ev.mask(c)
                if all(jaccard(mk, other) <= theta for other in masks):
                    beam.append(c)
                    masks.append(mk)
                    continue
            heapq.heappush(reserve, (-score(c), next(tie), c))
        return beam

    # 1. livello 1
    level = [frozenset([i]) for i in range(len(ev.items)) if add(frozenset([i]))]
    beam = select_diverse(level)
    stopped_by = "riserva vuota"
    while True:
        if over_budget():
            stopped_by = "budget"
            break
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
            beam = select_diverse(candidates)
        else:
            # 5. livello vuoto: si riparte dalla riserva
            if not reserve:
                break
            beam = [heapq.heappop(reserve)[2] for _ in range(min(beam_width, len(reserve)))]

    res = to_dataframe(ev, found)
    info = {"tempo": time.perf_counter() - t0, "valutazioni": ev.evaluations,
            "frequenti": len(found), "fermato_da": stopped_by}
    return res, ev, info
