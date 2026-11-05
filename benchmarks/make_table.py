"""Tabella dei risultati (Markdown) a partire dal CSV di un confronto.
Il file .md viene salvato accanto al CSV, nella stessa cartella.
Uso:
  python3 benchmarks/make_table.py benchmarks/results/small/ms0.05_K2000.csv
  python3 benchmarks/make_table.py benchmarks/results/medium/ms0.05_K2000.csv
  ...
"""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
t = pd.DataFrame({
    "Dataset": df["dataset"],
    "Algoritmo": df["algoritmo"],
    "Tempo (s)": df["tempo_s"].round(2),
    "Frequenti": df["frequenti"].astype(str) + "/" + df["frequenti_su"].astype(str),
    "Importanti": df["importanti"].astype(str) + "/" + df["importanti_su"].astype(str),
    "Recall T_min (%)": df["recall_%"].round(1),
    "Top-10 (%)": df["top10_%"].round(1),
    "Falsi": df["falsi"],
})
out = sys.argv[1].replace(".csv", ".md")
t.to_markdown(out, index=False)
print(t.to_markdown(index=False))
print(f"\nSalvata: {out}")