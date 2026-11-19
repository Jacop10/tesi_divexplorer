"""Algoritmo 3: Monte Carlo Tree Search (MCTS) con UCB.

Ogni nodo dell'albero e' un sottogruppo; i figli aggiungono una condizione.
A ogni iterazione:
  1. selezione: dalla radice si scende scegliendo il figlio con UCB piu' alto,
       UCB = M + c * sqrt(ln N_padre / N_figlio),
     dove M e' il premio MASSIMO visto passando da quel figlio (aggiornamento con il
     massimo: un ramo con pochi sottogruppi molto buoni non viene abbandonato);
  2. espansione: si aggiunge un figlio frequente nuovo. La prima volta che la
     selezione arriva a un nodo si valutano INSIEME tutti i suoi figli (un solo
     passaggio sulle sue righe, Evaluator.extensions);
  3. rollout GUIDATO DA UNA RIGA (come in Boley et al. e in random rows): si sceglie
     a caso la direzione (+/-) e una riga del nuovo nodo con probabilita'
     proporzionale al suo peso (booleano: outcome 1 per +, 0 per -; numerico:
     y - min per +, max - y per -); poi si aggiungono uno alla volta, in ordine
     casuale, i valori di quella riga sugli attributi non usati, finche' il
     sottogruppo resta frequente (o si arriva alla lunghezza massima);
  4. backpropagation: il premio (miglior |divergenza| significativa del rollout,
     normalizzata in [0, 1]) aggiorna visite e massimo di tutti i nodi del percorso.
Lo stesso sottogruppo raggiunto in ordini diversi e' UN SOLO nodo (duplicati
unificati). Tutti i sottogruppi frequenti visti vengono salvati.
Stop: pazienza (K iterazioni di fila senza sottogruppi importanti nuovi), albero
esaurito, oppure budget di tempo / iterazioni.
"""
import math
import time

import numpy as np

from .common import Evaluator, is_important, to_dataframe


class Node:
    __slots__ = ("itemset", "children", "untried", "N", "M", "dead")

    def __init__(self, itemset):
        self.itemset = itemset
        self.children = []       # nodi figli gia' nell'albero
        self.untried = None      # figli frequenti non ancora aggiunti (None = non espanso)
        self.N, self.M = 0, 0.0  # visite, premio massimo
        self.dead = False        # nessun discendente da esplorare


def mcts(df, attributes, outcome, kind, min_support, tau=2.0, eps=0.05, c=math.sqrt(2),
         K=5000, max_len=None, time_budget=None, max_iterations=None, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    ev = Evaluator(df, attributes, outcome, kind)
    max_len = max_len or len(attributes)
    scale = 1.0 if kind == "bool" else float(np.ptp(ev.y)) or 1.0
    y = ev.y
    weights = ({+1: (y == 1).astype(float), -1: (y == 0).astype(float)} if kind == "bool"
               else {+1: y - y.min(), -1: y.max() - y})
    found, found_at, nodes = {}, {}, {}
    new_important = False                    # trovato un importante in questa iterazione?

    def register(J, s):
        """Salva un sottogruppo frequente nuovo; True se e' frequente."""
        nonlocal new_important
        if s[0] / ev.n < min_support:
            return False
        if J not in found:
            found[J] = s
            found_at[J] = time.perf_counter() - t0
            if is_important(ev, J, tau, eps):
                new_important = True
        return True

    def node_of(J):
        if J not in nodes:
            nodes[J] = Node(J)
        return nodes[J]

    def expand_children(node):
        """Prima visita: tutti i figli frequenti valutati insieme."""
        if len(node.itemset) >= max_len:
            node.untried = []
            return
        kids = [node.itemset | {i} for i, s in ev.extensions(node.itemset).items()
                if register(node.itemset | {i}, s)]
        rng.shuffle(kids)
        node.untried = kids

    def reward(J):
        m = ev.measures(found[J])
        return abs(m["div"]) / scale if m["t"] >= tau else 0.0

    def rollout(J):
        """Rollout guidato da una riga del sottogruppo J."""
        best = reward(J)
        rows = ev.rows(J)
        if len(J) >= max_len or len(rows) == 0:
            return best
        d = +1 if rng.random() < 0.5 else -1
        w = weights[d][rows]
        r = rows[rng.choice(len(rows), p=w / w.sum())] if w.sum() > 0 else rng.choice(rows)
        used = {ev.item_attr[i] for i in J}
        free = [a for a in range(len(ev.attributes))
                if a not in used and ev.row_items[r, a] >= 0]
        rng.shuffle(free)
        for a in free:
            if len(J) >= max_len:
                break
            item = int(ev.row_items[r, a])
            J2 = J | {item}
            rows2 = rows[ev.row_items[rows, a] == item]
            s = ev.cache.get(J2)
            if s is None:
                s = ev.W[rows2].sum(axis=0)
                ev.cache[J2] = s
                ev.evaluations += 1
            if not register(J2, s):
                break
            J, rows = J2, rows2
            best = max(best, reward(J))
        return best

    root = node_of(frozenset())
    expand_children(root)
    iterations, since_new = 0, 0
    stopped_by = "albero esaurito"
    while not root.dead:
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

        # 1. selezione (si tiene il percorso per la backpropagation)
        path = [root]
        node = root
        while not node.untried and node.children:
            alive = [ch for ch in node.children if not ch.dead]
            if not alive:
                break
            parent = node
            node = max(alive, key=lambda ch: ch.M
                       + c * math.sqrt(math.log(max(parent.N, 1)) / max(ch.N, 1)))
            if node.untried is None:
                expand_children(node)
            path.append(node)
        # 2. espansione: un figlio non ancora nell'albero (duplicati unificati)
        child = None
        while node.untried:
            J = node.untried.pop()
            ch = node_of(J)
            if ch not in node.children:
                node.children.append(ch)
            child = ch           # i suoi figli si valutano solo quando verra' selezionato
            break
        if child is None:
            # nessun figlio da aggiungere: se non ha figli vivi, il nodo e' esaurito
            if not any(not ch.dead for ch in node.children):
                node.dead = True
                for p in reversed(path[:-1]):           # risale finche' serve
                    if p.untried or any(not ch.dead for ch in p.children):
                        break
                    p.dead = True
            since_new += 1
            continue
        path.append(child)
        # 3. rollout guidato da una riga
        best = rollout(child.itemset)
        # 4. backpropagation lungo il percorso seguito
        for p in path:
            p.N += 1
            p.M = max(p.M, best)
        since_new = 0 if new_important else since_new + 1

    res = to_dataframe(ev, found, found_at)
    info = {"tempo": time.perf_counter() - t0, "iterazioni": iterations,
            "valutazioni": ev.evaluations, "nodi": len(nodes), "frequenti": len(found),
            "fermato_da": stopped_by}
    return res, ev, info
