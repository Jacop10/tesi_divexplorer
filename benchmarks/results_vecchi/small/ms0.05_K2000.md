| Dataset         | Algoritmo   |   Tempo (s) |   Ultimo importante (s) | Frequenti   | Importanti   |   Recall T_min (%) |   Top-10 (%) |   Falsi |
|:----------------|:------------|------------:|------------------------:|:------------|:-------------|-------------------:|-------------:|--------:|
| compas_fpr      | DivExplorer |        0.05 |                    0.05 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | Random rows |        0.38 |                    0.02 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | Beam search |        0.05 |                    0.01 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | MCTS        |        0.14 |                    0.02 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fnr      | DivExplorer |        0.06 |                    0.06 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | Random rows |        0.37 |                    0.02 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | Beam search |        0.05 |                    0.03 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | MCTS        |        0.13 |                    0.04 | 315/315     | 36/36        |                100 |          100 |       0 |
| census          | DivExplorer |        0.53 |                    0.53 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | Random rows |        4.1  |                    0.11 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | Beam search |        0.98 |                    0.73 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | MCTS        |        1.07 |                    0.34 | 44/44       | 33/33        |                100 |          100 |       0 |
| synthetic_small | DivExplorer |        0.58 |                    0.58 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | Random rows |        1.42 |                    0.04 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | Beam search |        1.45 |                    1.17 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | MCTS        |        1.44 |                    1.23 | 1980/2084   | 94/94        |                100 |          100 |       0 |