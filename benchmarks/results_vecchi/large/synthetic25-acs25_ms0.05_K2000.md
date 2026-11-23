| Dataset       | Algoritmo   | Riferimento   |   Tempo (s) |   Ultimo importante (s) |   Memoria (GB) |   Frequenti | Importanti   |   Recall T_min (%) |   Top-10 (%) |   Falsi |
|:--------------|:------------|:--------------|------------:|------------------------:|---------------:|------------:|:-------------|-------------------:|-------------:|--------:|
| synthetic_A25 | DivExplorer | DivExplorer   |      256.76 |                  256.76 |           0.88 |      127456 | 114/114      |              100   |          100 |       0 |
| synthetic_A25 | Random rows | DivExplorer   |        6.28 |                    3.83 |           0.26 |      116515 | 114/114      |              100   |          100 |       0 |
| synthetic_A25 | Beam search | DivExplorer   |       14.58 |                    0.17 |           0.24 |        3977 | 38/114       |               33.3 |            0 |       0 |
| synthetic_A25 | MCTS        | DivExplorer   |        9.43 |                    8.5  |           0.2  |        8330 | 63/114       |               55.3 |            0 |       0 |
| acs_A25       | DivExplorer | DivExplorer   |       66.54 |                   66.54 |           5.8  |     4275614 | 2673/2673    |              100   |          100 |       0 |
| acs_A25       | Random rows | DivExplorer   |       50.66 |                   44.88 |           1    |      829974 | 2660/2673    |               99.5 |          100 |       0 |
| acs_A25       | Beam search | DivExplorer   |       30.51 |                   20.79 |           0.22 |        2870 | 171/2673     |                6.4 |          100 |       0 |
| acs_A25       | MCTS        | DivExplorer   |       16.32 |                   11.3  |           0.18 |        5496 | 417/2673     |               15.6 |           50 |       0 |