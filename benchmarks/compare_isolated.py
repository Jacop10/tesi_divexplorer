"""Confronto per i dataset grandi: ogni metodo gira in un PROCESSO SEPARATO.

Rispetto a compare.py:
- memoria: per ogni metodo si misura la memoria massima del suo processo (picco
  RSS, in GB). Comprende anche Python, le librerie e la copia del dataset;
- crash e tempi lunghi: se un metodo va in crash (es. memoria esaurita) o supera
  --timeout secondi, viene fermato e nella tabella compare "crash" o "timeout",
  e gli altri metodi continuano;
- riferimento: se DivExplorer finisce, la recall e' calcolata su T_min (esatta).
  Se DivExplorer non finisce (o lo si salta con --skip-divexplorer), il
  riferimento e' l'UNIONE dei sottogruppi importanti trovati dai metodi
  approssimati: ogni sottogruppo trovato e' davvero importante (i valori sono
  esatti), quindi l'unione e' un sottoinsieme di T_min e la recall rispetto
  all'unione e' una stima per eccesso.
Il processo figlio restituisce solo i sottogruppi importanti (non tutti i
frequenti), cosi' anche con milioni di frequenti il passaggio dei dati e' leggero.
"""
import multiprocessing as mp
import os
import queue
import resource
import sys
import time

import pandas as pd

from compare import HERE, add_args, run_divexplorer

METHODS = ["DivExplorer", "Random rows", "Beam search", "MCTS"]
KEY = {"random": "Random rows", "beam": "Beam search", "mcts": "MCTS"}


def _peak_memory_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)   # macOS: byte, Linux: KB


def _run_method(method, df, attrs, outcome, kind, p):
    """Gira nel processo figlio. Restituisce tempo, numero di frequenti e, per i soli
    sottogruppi importanti, divergenza e momento in cui sono stati trovati."""
    from divexplorer.approximate_algorithms.beam_search import beam_search
    from divexplorer.approximate_algorithms.common import Evaluator, important_itemsets
    from divexplorer.approximate_algorithms.mcts import mcts
    from divexplorer.approximate_algorithms.random_rows import random_rows

    if method == "DivExplorer":
        res, t = run_divexplorer(df, attrs, outcome, kind, p["min_support"])
        info = {"tempo": t}
        ev = Evaluator(df, attrs, outcome, kind)
    elif method == "Random rows":
        res, ev, info = random_rows(df, attrs, outcome, kind, p["min_support"], p["tau"],
                                    p["eps"], m=p["m"], K=p["K"], evaluation=p["evaluation"],
                                    output=p["output"],
                                    seed=p["seed"])
    elif method == "Beam search":
        res, ev, info = beam_search(df, attrs, outcome, kind, p["min_support"], p["tau"],
                                    p["eps"], beam_width=p["beam_width"], theta=p["theta"],
                                    K=p["K_beam"], time_budget=p["budget"])
    else:
        res, ev, info = mcts(df, attrs, outcome, kind, p["min_support"], p["tau"], p["eps"],
                             c=p["ucb_c"], K=p["K_mcts"], time_budget=p["budget"],
                             seed=p["seed"])
    if info.get("output") == "important":  # random rows ha gia' tenuto solo gli importanti
        T = set(res["itemset"])
    else:
        T = important_itemsets(res, outcome, p["tau"], p["eps"], ev) if len(res) else set()
    imp = res[res["itemset"].isin(T)]
    t_found = (dict(zip(imp["itemset"], imp["t_found"])) if "t_found" in imp.columns
               else {i: info["tempo"] for i in imp["itemset"]})
    return {"tempo": info["tempo"],
            "frequenti": len(res) if info.get("output") != "important" else float("nan"),
            "div": dict(zip(imp["itemset"], imp[f"{outcome}_div"])), "t_found": t_found,
            "note": ", ".join(f"{k}={v}" for k, v in info.items() if k != "tempo")}


def _worker(q, method, df, attrs, outcome, kind, p):
    try:
        out = _run_method(method, df, attrs, outcome, kind, p)
        out["mem_gb"] = _peak_memory_gb()
        q.put(out)
    except MemoryError:
        q.put({"errore": "memoria esaurita"})
    except Exception as e:                       # noqa: BLE001
        q.put({"errore": f"{type(e).__name__}: {e}"})


def run_isolated(method, df, attrs, outcome, kind, p, timeout):
    """Lancia un metodo in un processo separato; si ferma dopo `timeout` secondi."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(q, method, df, attrs, outcome, kind, p))
    t0 = time.perf_counter()
    proc.start()
    out = None
    while out is None:
        try:
            out = q.get(timeout=2)
        except queue.Empty:
            if time.perf_counter() - t0 > timeout:
                proc.kill()
                out = {"errore": f"timeout ({timeout:.0f} s)"}
            elif not proc.is_alive():
                try:
                    out = q.get(timeout=2)
                except queue.Empty:
                    out = {"errore": f"crash (codice {proc.exitcode}, memoria esaurita?)"}
    proc.join(timeout=10)
    return out


def main(group, loaders, doc):
    import argparse
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p, loaders)
    p.add_argument("--timeout", type=float, default=1800,
                   help="secondi massimi per ogni metodo (default 1800 = 30 minuti)")
    p.add_argument("--skip-divexplorer", action="store_true",
                   help="non lanciare DivExplorer (riferimento = unione dei metodi)")
    args = p.parse_args()

    rows = []
    for ds in args.datasets:
        df, attrs, outcome, kind, name = loaders[ds]()
        eps = args.eps if args.eps is not None else (
            0.05 if kind == "bool" else 0.01 * float(df[outcome].std()))
        print(f"\n=== {name}: {len(df)} righe, {len(attrs)} attributi, outcome {outcome} "
              f"({kind}), min_support={args.min_support}, tau={args.tau}, eps={eps:.4g}",
              flush=True)
        params = {"min_support": args.min_support, "tau": args.tau, "eps": eps, "m": args.m,
                  "K": args.K, "K_beam": args.K_beam, "K_mcts": args.K_mcts,
                  "evaluation": args.evaluation, "output": args.output, "beam_width": args.beam_width,
                  "theta": args.theta, "ucb_c": args.ucb_c, "seed": args.seed,
                  "budget": args.budget or 600.0}
        methods = ([] if args.skip_divexplorer else ["DivExplorer"]) + \
                  [KEY[a] for a in ["random", "beam", "mcts"] if a in args.algorithms]
        outs = {}
        for method in methods:
            out = run_isolated(method, df, attrs, outcome, kind, params, args.timeout)
            outs[method] = out
            if "errore" in out:
                print(f"{method:12s}: {out['errore']}", flush=True)
                continue
            print(f"{method:12s}: {out['tempo']:.1f} s, {out['mem_gb']:.2f} GB, "
                  f"{out['frequenti']} frequenti, {len(out['div'])} importanti", flush=True)
            if method == "Random rows" and not args.budget:
                params["budget"] = 10 * out["tempo"]      # limite di sicurezza per beam/MCTS

        # riferimento: T_min di DivExplorer, oppure unione dei metodi
        exact = "DivExplorer" in outs and "errore" not in outs["DivExplorer"]
        if exact:
            ref_div = outs["DivExplorer"]["div"]
        else:
            ref_div = {}
            for out in outs.values():
                if "errore" not in out:
                    ref_div.update(out["div"])
        T_ref = set(ref_div)
        top10 = set(sorted(T_ref, key=lambda i: abs(ref_div[i]), reverse=True)[:10])
        print(f"(riferimento: {'T_min di DivExplorer' if exact else 'unione dei metodi'}, "
              f"{len(T_ref)} sottogruppi importanti)", flush=True)

        for method, out in outs.items():
            row = {"dataset": name, "algoritmo": method,
                   "riferimento": "DivExplorer" if exact else "unione"}
            if "errore" in out:
                row["note"] = out["errore"]
                rows.append(row)
                continue
            T = set(out["div"])
            found = T & T_ref
            row.update({"tempo_s": out["tempo"],
                        "t_ultimo_s": max((out["t_found"][i] for i in found), default=float("nan")),
                        "mem_GB": out["mem_gb"], "frequenti": out["frequenti"],
                        "importanti": len(found), "importanti_su": len(T_ref),
                        "recall_%": 100 * len(found) / max(len(T_ref), 1),
                        "top10_%": 100 * len(found & top10) / max(len(top10), 1),
                        "falsi": len(T - T_ref) if exact else float("nan"),
                        "note": out["note"]})
            rows.append(row)

    table = pd.DataFrame(rows)
    cols = ["dataset", "algoritmo", "riferimento", "tempo_s", "t_ultimo_s", "mem_GB",
            "frequenti", "importanti", "importanti_su", "recall_%", "top10_%", "falsi", "note"]
    table = table.reindex(columns=cols)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + table.drop(columns="note").round(3).to_string(index=False))
    print("\nNote:\n" + "\n".join(f"  {r.dataset} / {r.algoritmo}: {r.note}"
                                   for r in table.itertuples() if isinstance(r.note, str)))
    out_dir = os.path.join(HERE, "results", group)
    os.makedirs(out_dir, exist_ok=True)
    tag = "-".join(args.datasets)
    out = os.path.join(out_dir, f"{tag}_ms{args.min_support}_K{args.K}.csv")
    table.to_csv(out, index=False)
    print(f"\nSalvato in {out}")
