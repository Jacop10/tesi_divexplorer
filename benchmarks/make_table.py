"""Tabella dei risultati (Markdown) a partire dal CSV di un confronto.
Il file .md viene salvato accanto al CSV, nella stessa cartella.
Funziona per small, medium e large (le colonne che mancano vengono saltate).
Uso:
  python3 benchmarks/make_table.py benchmarks/results/small/ms0.05_K2000.csv
  python3 benchmarks/make_table.py benchmarks/results/large/acs25_ms0.05_K2000.csv
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])


def num(col, digits=2):
    return df[col].round(digits) if col in df.columns else None


def ratio(col, total):
    if col not in df.columns:
        return None
    if total not in df.columns:
        return df[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return [("" if pd.isna(a) else f"{int(a)}/{int(b)}") for a, b in zip(df[col], df[total])]


columns = {
    "Dataset": df["dataset"],
    "Algoritmo": df["algoritmo"],
    "Riferimento": df["riferimento"] if "riferimento" in df.columns else None,
    "Tempo (s)": num("tempo_s"),
    "Ultimo importante (s)": num("t_ultimo_s"),
    "Memoria (GB)": num("mem_GB"),
    "Frequenti": ratio("frequenti", "frequenti_su"),
    "Importanti": ratio("importanti", "importanti_su"),
    "Recall T_min (%)": num("recall_%", 1),
    "Top-10 (%)": num("top10_%", 1),
    "Falsi": df["falsi"].map(lambda v: "" if pd.isna(v) else int(v)) if "falsi" in df.columns else None,
}
t = pd.DataFrame({k: v for k, v in columns.items() if v is not None})
if "note" in df.columns and "tempo_s" in df.columns:      # errori (crash, timeout)
    errors = df["tempo_s"].isna() & df["note"].notna()
    if errors.any():
        t["Tempo (s)"] = t["Tempo (s)"].astype(object)
        t.loc[errors, "Tempo (s)"] = df.loc[errors, "note"]
t = t.fillna("")
out = sys.argv[1].replace(".csv", ".md")
t.to_markdown(out, index=False)
print(t.to_markdown(index=False))
print(f"\nSalvata: {out}")
