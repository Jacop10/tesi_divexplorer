| Dataset         | Algoritmo   |   Tempo (s) | Frequenti   | Importanti   |   Recall T_min (%) |   Top-10 (%) |   Falsi |
|:----------------|:------------|------------:|:------------|:-------------|-------------------:|-------------:|--------:|
| compas_fpr      | DivExplorer |        0.05 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | Random rows |        0.97 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | Beam search |        0.05 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fpr      | MCTS        |        0.97 | 315/315     | 10/10        |                100 |          100 |       0 |
| compas_fnr      | DivExplorer |        0.04 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | Random rows |        0.99 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | Beam search |        0.05 | 315/315     | 36/36        |                100 |          100 |       0 |
| compas_fnr      | MCTS        |        0.99 | 315/315     | 36/36        |                100 |          100 |       0 |
| census          | DivExplorer |        0.53 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | Random rows |        9.35 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | Beam search |        0.94 | 44/44       | 33/33        |                100 |          100 |       0 |
| census          | MCTS        |        9.35 | 44/44       | 33/33        |                100 |          100 |       0 |
| synthetic_small | DivExplorer |        0.61 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | Random rows |        5.26 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | Beam search |        1.36 | 2084/2084   | 94/94        |                100 |          100 |       0 |
| synthetic_small | MCTS        |        5.27 | 2073/2084   | 94/94        |                100 |          100 |       0 |