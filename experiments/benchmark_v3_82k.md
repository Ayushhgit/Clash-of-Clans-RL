# Benchmark (privileged)

| agent | L2 | L3 | L4 |
|---|---|---|---|
| random | 95% | 55% | 10% |
| heuristic | 100% | 85% | 20% |
| ppo | 80% | 35% | 10% |

random       L2  win  0.95  dest 0.790  loot 0.856  R   5.143  eff 0.529  t  149.6s   0.01 ms/act
random       L3  win  0.55  dest 0.463  loot 0.547  R   2.103  eff 0.290  t  180.1s   0.01 ms/act
random       L4  win  0.10  dest 0.197  loot 0.242  R  -0.335  eff 0.126  t  173.4s   0.01 ms/act
heuristic    L2  win  1.00  dest 0.748  loot 0.862  R   5.074  eff 0.523  t  153.7s   0.02 ms/act
heuristic    L3  win  0.85  dest 0.444  loot 0.650  R   3.221  eff 0.245  t  170.0s   0.02 ms/act
heuristic    L4  win  0.20  dest 0.241  loot 0.474  R   0.233  eff 0.133  t  152.3s   0.02 ms/act
ppo          L2  win  0.80  dest 0.745  loot 0.828  R   4.350  eff 0.448  t  150.6s  11.11 ms/act
ppo          L3  win  0.35  dest 0.414  loot 0.572  R   1.285  eff 0.235  t  164.9s  12.21 ms/act
ppo          L4  win  0.10  dest 0.225  loot 0.271  R  -0.357  eff 0.127  t  150.8s  12.13 ms/act
