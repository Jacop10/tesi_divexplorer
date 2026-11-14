"""Confronto su dataset grandi: DivExplorer (originale) e i tre algoritmi approssimati.

Ogni metodo gira in un processo separato: si misura la memoria massima, e un crash o
un tempo troppo lungo (--timeout) ferma solo quel metodo. Se DivExplorer non finisce,
la recall e' calcolata rispetto all'unione dei sottogruppi importanti trovati dai
metodi. Dettagli in compare_isolated.py. Risultati in benchmarks/results/large/,
con i nomi dei dataset nel nome del file (prove diverse non si sovrascrivono).

Esempi (dalla cartella principale del repository):
  python3 benchmarks/run_large.py --datasets synthetic25 acs25
  python3 benchmarks/run_large.py --datasets acs30 acs35 --timeout 3600
  python3 benchmarks/run_large.py --datasets acs35 --skip-divexplorer
  python3 benchmarks/run_large.py --datasets acs20 --min-support 0.01
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_isolated import main   # noqa: E402
from datasets_large import LOADERS  # noqa: E402

if __name__ == "__main__":
    main("large", LOADERS, __doc__)
