"""Algoritmo 3: Monte Carlo Tree Search (MCTS) con UCB.

Ogni nodo dell'albero e' un sottogruppo; i figli aggiungono una condizione.
A ogni iterazione:
  1. selezione: dalla radice si scende scegliendo il figlio con UCB piu' alto,
       UCB = Q/N + c * sqrt(ln N_padre / N_figlio),
     finche' si trova un nodo con figli ancora da provare;
  2. espansione: si aggiunge un figlio frequente nuovo;
  3. rollout: dal figlio si aggiungono condizioni a caso finche' il sottogruppo
     resta frequente (o si arriva alla lunghezza massima);
  4. backpropagation: il premio (miglior |divergenza| significativa del rollout,
     normalizzata in [0, 1]) si somma a Q di tutti i nodi del percorso.
Tutti i sottogruppi frequenti visti vengono salvati.
Stop: pazienza (K iterazioni di fila senza sottogruppi importanti nuovi), albero
esaurito, oppure budget di tempo / iterazioni.
"""
import math
import time

import numpy as np

from .common import Evaluator, is_important, to_dataframe


class Node:
    def __init__(self, itemset, parent, untried):
        self.itemset, self.parent = itemset, parent
        self.children, self.untried = [], untried
        self.N, self.Q = 0, 0.0


def mcts(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05, c=math.sqrt(2),
         K=5000, max_len=None, time_budget=None, max_iterations=None, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    ev = Evaluator(df, attributes, outcome, kind)
    max_len = max_len or len(attributes)
    scale = 1.0 if kind == "bool" else float(np.ptp(ev.y)) or 1.0
    found, found_at = {}, {}
    new_important = False                    # trovato un importante in questa iterazione?

    def frequent(itemset):
        nonlocal new_important
        if itemset in found:
            return True
        s = ev.stats(itemset)
        if s[0] / ev.n < min_support:
            return False
        found[itemset] = s
        found_at[itemset] = time.perf_counter() - t0
        if is_important(ev, itemset, tau, eps):
            new_important = True
        return True

    def extensions(itemset):
        if len(itemset) >= max_len:
            return []
        used = {ev.item_attr[i] for i in itemset}
        ext = [i for i in range(len(ev.items)) if ev.item_attr[i] not in used]
        rng.shuffle(ext)
        return ext

    def reward(itemset):
        m = ev.measures(found[itemset])
        return abs(m["div"]) / scale if m["t"] >= tau else 0.0

    root = Node(frozenset(), None, extensions(frozenset()))
    iterations, since_new = 0, 0
    stopped_by = "albero esaurito"
    while True:
        if time_budget is not None and time.perf_counter() - t0 > time_budget:
            stopped_by = "tempo"
            break
        if max_iterations is not None and iterations >= max_iterations:
            stopped_by = "max_iterazioni"
            break
        if since_new >= K:
            stopped_by = "pazienza"
            break
        iterations += 1
        new_important = False

        # 1. selezione
        node = root
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.Q / ch.N
                       + c * math.sqrt(math.log(node.N) / ch.N))
        # 2. espansione (primo figlio frequente tra quelli non provati)
        child = None
        while node.untried and child is None:
            J = node.itemset | {node.untried.pop()}
            if frequent(J):
                child = Node(J, node, extensions(J))
                node.children.append(child)
        if child is None:                    # nessun figlio frequente nuovo
            if node is root:
                if not root.children:        # nessun item frequente: fine
                    break
                since_new += 1
                continue                     # la radice ha finito i figli: si scende
            child = node                     # foglia: si rifa' il rollout da qui
        # 3. rollout casuale
        best, J = reward(child.itemset), child.itemset
        while len(J) < max_len:
            ext = extensions(J)
            if not ext:
                break
            J2 = J | {ext[0]}
            if not frequent(J2):
                break
            J = J2
            best = max(best, reward(J))
        # 4. backpropagation
        while child is not None:
            child.N += 1
            child.Q += best
            child = child.parent
        since_new = 0 if new_important else since_new + 1

    res = to_dataframe(ev, found, found_at)
    info = {"tempo": time.perf_counter() - t0, "iterazioni": iterations,
            "valutazioni": ev.evaluations, "frequenti": len(found),
            "fermato_da": stopped_by}
    return res, ev, info
