"""Confronto su dataset piccoli: DivExplorer (originale) e i tre algoritmi approssimati.

Dataset: COMPAS FPR e FNR (6 attributi), census (4 attributi, reddito numerico),
sintetico con 10 attributi. Il confronto e' descritto in compare.py.
Risultati in benchmarks/results/small/.

Esempi (dalla cartella principale del repository):
  python3 benchmarks/run_small.py
  python3 benchmarks/run_small.py --datasets compas_fpr census --K 2000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare import main            # noqa: E402
from datasets_small import LOADERS  # noqa: E402

if __name__ == "__main__":
    main("small", LOADERS, __doc__)
