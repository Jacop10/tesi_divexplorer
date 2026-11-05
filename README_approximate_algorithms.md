# Tesi: algoritmi approssimati per DivExplorer

<a href="https://colab.research.google.com/github/Jacop10/tesi_divexplorer/blob/main/notebooks/ApproximateAlgorithmsExample.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Tesi di Jacopo Santolupo (Politecnico di Torino), relatrice Prof.ssa Eliana Pastor.

[DivExplorer](https://github.com/divexplorer/divexplorer) trova i sottogruppi di dati
*divergenti*, cioè quelli in cui un outcome (per esempio il false positive rate di un
classificatore o il reddito medio) è molto diverso da quello del dataset intero. È
**esatto e completo**: calcola tutti i sottogruppi frequenti, ma il loro numero cresce
in modo esponenziale con gli attributi.

Questo repository parte dal codice di DivExplorer e aggiunge tre algoritmi
**approssimati**. Non garantiscono di trovare tutti i sottogruppi (non completi), ma i
valori che trovano (supporto, divergenza, t-value) sono esatti (corretti):

| Algoritmo | Idea |
|---|---|
| **Random rows** | Si sceglie a caso una riga e m attributi; con la *matrice compatta* si calcolano in un colpo tutti i sottogruppi che contengono quella riga. Si ferma dopo K campioni di fila senza sottogruppi importanti nuovi (pazienza). |
| **Beam search** (modificata) | Si estendono livello per livello solo i w sottogruppi migliori e diversi tra loro; gli scartati vanno in una riserva da cui si riparte quando un livello resta vuoto. |
| **MCTS** | Monte Carlo Tree Search: si scende nell'albero dei sottogruppi scegliendo con UCB, si estende a caso e si premiano i percorsi con divergenza alta. |

## Struttura

I file aggiunti rispetto a DivExplorer sono segnati con ★.

```
tesi_divexplorer/
├── datasets/                         dataset di DivExplorer (COMPAS, census)
├── divexplorer/
│   ├── divexplorer.py                DivExplorer originale
│   ├── ...
│   └── approximate_algorithms/       ★ algoritmi approssimati
│       ├── common.py                 formule (come DivExplorer), sottogruppi importanti T_min
│       ├── random_rows.py            Algoritmo 1
│       ├── beam_search.py            Algoritmo 2
│       └── mcts.py                   Algoritmo 3
├── benchmarks/                       ★
│   ├── datasets_small.py             caricamento dei dataset piccoli
│   ├── run_small.py                  confronto DivExplorer vs i 3 algoritmi
│   ├── make_table.py                 tabella Markdown dai risultati
│   └── results/                      risultati, una cartella per gruppo di prove
│       └── small/
└── notebooks/
    ├── DivExplorerExample.ipynb      esempio originale di DivExplorer
    └── ApproximateAlgorithmsExample.ipynb   ★ stesso esempio con i 4 metodi
```

## Installazione

```
git clone https://github.com/Jacop10/tesi_divexplorer.git
cd tesi_divexplorer
pip install -r requirements.txt
```

Tutti i comandi seguenti vanno lanciati dalla cartella principale `tesi_divexplorer/`.
Gli script usano il DivExplorer contenuto nel repository, quindi non serve
`pip install divexplorer`.

## 1. Confronto sui dataset piccoli

```
python3 benchmarks/run_small.py --K 2000
```

Per ogni dataset (COMPAS FPR, COMPAS FNR, census, sintetico con 10 attributi):

1. DivExplorer calcola tutti i sottogruppi frequenti: è il riferimento esatto;
2. si calcola T_min, l'insieme dei sottogruppi **importanti**: frequenti, con t ≥ τ e
   non ridondanti;
3. random rows gira finché non scatta la pazienza K;
4. beam search e MCTS ricevono **lo stesso tempo** di random rows.

Opzioni principali:

| Opzione | Default | Significato |
|---|---|---|
| `--datasets` | tutti | `compas_fpr`, `compas_fnr`, `census`, `synthetic` |
| `--min-support` | 0.05 | supporto minimo |
| `--tau` | 2.0 | soglia sul t-value |
| `--eps` | 0.05 (bool), 0.01·std (num) | soglia di ridondanza |
| `--K` | 2000 | pazienza di random rows |
| `--m` | 10 | attributi per campione |
| `--beam-width` | 10 | ampiezza del beam |
| `--theta` | 0.5 | sovrapposizione massima (Jaccard) nel beam |
| `--ucb-c` | √2 | costante di esplorazione di UCB |
| `--budget` | tempo di random rows | secondi per beam e MCTS |

Esempio su due soli dataset:
```
python3 benchmarks/run_small.py --datasets compas_fpr census --K 2000
```

Il risultato viene salvato in `benchmarks/results/small/ms<min_support>_K<K>.csv`.

### Colonne dei risultati

| Colonna | Significato |
|---|---|
| `tempo_s` | secondi |
| `frequenti` / `frequenti_su` | sottogruppi frequenti trovati / totali (DivExplorer) |
| `importanti` / `importanti_su` | sottogruppi di T_min trovati / totali |
| `recall_%` | percentuale di T_min trovata |
| `top10_%` | percentuale dei 10 sottogruppi di T_min più divergenti trovati |
| `falsi` | sottogruppi dichiarati importanti ma non in T_min (deve essere 0) |
| `max_diff_div` | massima differenza di divergenza da DivExplorer (deve essere 0) |

## 2. Tabella dei risultati

```
pip install tabulate
python3 benchmarks/make_table.py benchmarks/results/small/ms0.05_K2000.csv
```

Crea `ms0.05_K2000.md` accanto al CSV, pronto da incollare in un documento.

## 3. Notebook: quali sottogruppi trova ogni metodo

`notebooks/ApproximateAlgorithmsExample.ipynb` segue lo stesso percorso del notebook
originale di DivExplorer (census e COMPAS) e mostra per ogni metodo:

- i sottogruppi più divergenti;
- per i sottogruppi più importanti, quali metodi li hanno trovati (✓/✗);
- redundancy pruning, Shapley value locale e globale.

**In locale:** aprilo in VS Code (serve l'estensione Jupyter) oppure con
```
pip install notebook
jupyter notebook notebooks/ApproximateAlgorithmsExample.ipynb
```

**Su Colab:** clicca il pulsante *Open in Colab* in cima a questa pagina. La prima cella
scarica il repository e installa le librerie da sola. Colab legge il ramo `main`.

## 4. Usare gli algoritmi nel proprio codice

Restituiscono un DataFrame nello stesso formato di DivExplorer (`support`, `itemset`,
outcome, `<outcome>_div`, `<outcome>_t`, `length`):

```python
import pandas as pd
from divexplorer.outcomes import get_false_positive_rate_outcome
from divexplorer.approximate_algorithms.random_rows import random_rows
from divexplorer.approximate_algorithms.beam_search import beam_search
from divexplorer.approximate_algorithms.mcts import mcts

df = pd.read_csv("datasets/compas_discretized.csv")
df["fp"] = get_false_positive_rate_outcome(df["class"], df["predicted"])
attrs = ["age", "charge", "race", "sex", "#prior", "stay"]

# kind = "bool" per outcome booleani (1 / 0 / NaN), "num" per outcome numerici
res, ev, info = random_rows(df, attrs, "fp", "bool", min_support=0.05, K=2000)
res.sort_values("fp_div", ascending=False).head(10)

res, ev, info = beam_search(df, attrs, "fp", "bool", min_support=0.05, time_budget=5)
res, ev, info = mcts(df, attrs, "fp", "bool", min_support=0.05, time_budget=5)
```

`info` contiene tempo, numero di campioni / valutazioni / iterazioni e il motivo dello stop.

## Lavorare sul repository

Si lavora sul ramo `dev` e si porta tutto su `main` alla fine:

```
git checkout dev                    # lavoro
git add . && git commit -m "..." && git push
git checkout main && git merge dev && git push   # alla fine
```

## Riferimenti

- E. Pastor, L. de Alfaro, E. Baralis. *Looking for Trouble: Analyzing Classifier
  Behavior via Pattern Divergence.* SIGMOD 2021. (DivExplorer)
- M. Boley, C. Lucchese, D. Paurat, T. Gärtner. *Direct Local Pattern Sampling by
  Efficient Two-Step Random Procedures.* KDD 2011. (campionamento di pattern)
- L. Kocsis, C. Szepesvári. *Bandit Based Monte-Carlo Planning.* ECML 2006. (UCT)
- G. Bosc, J.-F. Boulicaut, C. Raïssi, M. Kaytoue. *Anytime Discovery of a Diverse Set
  of Patterns with Monte Carlo Tree Search.* DMKD 2018. (MCTS per i sottogruppi)

Il README originale di DivExplorer è in [`README_DivExplorer.md`](README_DivExplorer.md).