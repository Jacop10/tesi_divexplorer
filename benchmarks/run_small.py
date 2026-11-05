"""Confronto su dataset piccoli: DivExplorer (originale) e i tre algoritmi approssimati.

Per ogni dataset:
  - DivExplorer calcola TUTTI i sottogruppi frequenti -> riferimento esatto;
  - T_min = sottogruppi frequenti, con t >= tau e non ridondanti;
  - random rows si ferma con la pazienza K;
  - beam search e MCTS ricevono lo STESSO tempo di random rows (budget uguale).
Si stampa: tempo, sottogruppi frequenti trovati, recall su T_min, recall sui 10
sottogruppi di T_min con |divergenza| piu' alta, e la massima differenza di
divergenza rispetto a DivExplorer (deve essere ~0: i valori sono esatti).

Esempi (dalla cartella principale del repository):
  python3 benchmarks/run_small.py
  python3 benchmarks/run_small.py --datasets compas_fpr census --min-support 0.05 --K 2000
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # cartella principale del repository
sys.path.insert(0, HERE)

from divexplorer import DivergenceExplorer                       # noqa: E402
from divexplorer.approximate_algorithms.beam_search import beam_search           # noqa: E402
from divexplorer.approximate_algorithms.common import Evaluator, important_itemsets  # noqa: E402
from divexplorer.approximate_algorithms.mcts import mcts                         # noqa: E402
from divexplorer.approximate_algorithms.random_rows import random_rows           # noqa: E402
from datasets_small import LOADERS                               # noqa: E402


def run_divexplorer(df, attrs, outcome, kind, min_support):
    t = time.perf_counter()
    explorer = DivergenceExplorer(df)
    args = {"boolean_outcomes": [outcome]} if kind == "bool" else \
           {"quantitative_outcomes": [outcome]}
    res = explorer.get_pattern_divergence(min_support=min_support, attributes=attrs, **args)
    res = res[res["length"] > 0].reset_index(drop=True)
    return res, time.perf_counter() - t


def compare(name, res, ev, info, ref, T_true, top10, outcome, tau, eps):
    T = important_itemsets(res, outcome, tau, eps, ev) if len(res) else set()
    ref_div = dict(zip(ref["itemset"], ref[f"{outcome}_div"]))
    diffs = [abs(d - ref_div[i]) for i, d in zip(res["itemset"], res[f"{outcome}_div"])
             if i in ref_div]
    return {"algoritmo": name, "tempo_s": info["tempo"],
            "frequenti": len(res), "frequenti_su": len(ref),
            "importanti": len(T & T_true), "importanti_su": len(T_true),
            "recall_%": 100 * len(T & T_true) / max(len(T_true), 1),
            "top10_%": 100 * len(T & top10) / max(len(top10), 1),
            "falsi": len(T - T_true),
            "max_diff_div": max(diffs) if diffs else 0.0,
            "note": ", ".join(f"{k}={v}" for k, v in info.items() if k != "tempo")}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=2.0, help="soglia sul t-value")
    p.add_argument("--eps", type=float, default=None,
                   help="soglia di ridondanza (default 0.05 booleani, 0.01*std numerici)")
    p.add_argument("--K", type=int, default=2000, help="pazienza di random rows")
    p.add_argument("--m", type=int, default=10, help="attributi per campione")
    p.add_argument("--beam-width", type=int, default=10)
    p.add_argument("--theta", type=float, default=0.5, help="Jaccard massimo nel beam")
    p.add_argument("--ucb-c", type=float, default=2 ** 0.5)
    p.add_argument("--budget", type=float, default=None,
                   help="secondi per beam e MCTS (default: tempo di random rows)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = []
    for ds in args.datasets:
        df, attrs, outcome, kind, name = LOADERS[ds]()
        eps = args.eps if args.eps is not None else (
            0.05 if kind == "bool" else 0.01 * float(df[outcome].std()))
        print(f"\n=== {name}: {len(df)} righe, {len(attrs)} attributi, outcome {outcome} "
              f"({kind}), min_support={args.min_support}, tau={args.tau}, eps={eps:.4g}",
              flush=True)

        ref, t_dx = run_divexplorer(df, attrs, outcome, kind, args.min_support)
        ev_ref = Evaluator(df, attrs, outcome, kind)
        T_true = important_itemsets(ref, outcome, args.tau, eps, ev_ref)
        imp = ref[ref["itemset"].isin(T_true)]
        top10 = set(imp.reindex(imp[f"{outcome}_div"].abs()
                                .sort_values(ascending=False).index)["itemset"][:10])
        base = {"tempo": t_dx}
        rows.append(compare("DivExplorer", ref, ev_ref, base, ref, T_true, top10,
                            outcome, args.tau, eps) | {"dataset": name})
        print(f"DivExplorer: {t_dx:.2f} s, {len(ref)} frequenti, {len(T_true)} in T_min",
              flush=True)

        res, ev, info = random_rows(df, attrs, outcome, kind, args.min_support, args.tau,
                                    eps, m=args.m, K=args.K, seed=args.seed)
        rows.append(compare("Random rows", res, ev, info, ref, T_true, top10,
                            outcome, args.tau, eps) | {"dataset": name})
        print(f"Random rows: {info['tempo']:.2f} s", flush=True)

        budget = args.budget or info["tempo"]
        res, ev, info = beam_search(df, attrs, outcome, kind, args.min_support,
                                    beam_width=args.beam_width, theta=args.theta,
                                    time_budget=budget)
        rows.append(compare("Beam search", res, ev, info, ref, T_true, top10,
                            outcome, args.tau, eps) | {"dataset": name})
        print(f"Beam search: {info['tempo']:.2f} s", flush=True)

        res, ev, info = mcts(df, attrs, outcome, kind, args.min_support, args.tau,
                             c=args.ucb_c, time_budget=budget, seed=args.seed)
        rows.append(compare("MCTS", res, ev, info, ref, T_true, top10,
                            outcome, args.tau, eps) | {"dataset": name})
        print(f"MCTS:        {info['tempo']:.2f} s", flush=True)

    table = pd.DataFrame(rows)
    cols = ["dataset", "algoritmo", "tempo_s", "frequenti", "frequenti_su", "importanti",
            "importanti_su", "recall_%", "top10_%", "falsi", "max_diff_div", "note"]
    table = table[cols]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + table.drop(columns="note").round(3).to_string(index=False))
    print("\nNote:\n" + "\n".join(f"  {r.dataset} / {r.algoritmo}: {r.note}"
                                   for r in table.itertuples() if r.algoritmo != "DivExplorer"))
    out_dir = os.path.join(HERE, "results", "small")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"ms{args.min_support}_K{args.K}.csv")
    table.to_csv(out, index=False)
    print(f"\nSalvato in {out}")


if __name__ == "__main__":
    main()
