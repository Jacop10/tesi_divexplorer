"""Confronto su dataset medi: DivExplorer (originale) e i tre algoritmi approssimati.

Dataset: sintetico con 15 e 20 attributi, ACS con 15 e 20 attributi.
Il confronto e' descritto in compare.py. Risultati in benchmarks/results/medium/.

Esempi (dalla cartella principale del repository):
  python3 benchmarks/run_medium.py
  python3 benchmarks/run_medium.py --datasets synthetic15 acs15
  python3 benchmarks/run_medium.py --datasets synthetic20 --K-beam 1000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare import main             # noqa: E402
from datasets_medium import LOADERS  # noqa: E402

if __name__ == "__main__":
    main("medium", LOADERS, __doc__)
