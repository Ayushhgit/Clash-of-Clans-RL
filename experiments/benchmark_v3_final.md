# Benchmark (privileged)

| agent | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| random | 85% | 38% | 2% | 5% |
| heuristic | 100% | 70% | 12% | 22% |
| ppo | 100% | 62% | 12% | 10% |

random       L2  win  0.85  dest 0.738  loot 0.834  R   4.584  eff 0.503  t  147.6s   0.01 ms/act
random       L3  win  0.38  dest 0.434  loot 0.520  R   1.351  eff 0.265  t  173.6s   0.01 ms/act
random       L4  win  0.03  dest 0.211  loot 0.235  R  -0.662  eff 0.123  t  159.4s   0.01 ms/act
random       L5  win  0.05  dest 0.115  loot 0.103  R  -0.821  eff 0.064  t  158.0s   0.01 ms/act
heuristic    L2  win  1.00  dest 0.824  loot 0.911  R   5.487  eff 0.573  t  145.5s   0.03 ms/act
heuristic    L3  win  0.70  dest 0.422  loot 0.643  R   2.674  eff 0.238  t  153.9s   0.03 ms/act
heuristic    L4  win  0.12  dest 0.228  loot 0.392  R  -0.131  eff 0.126  t  143.3s   0.04 ms/act
heuristic    L5  win  0.23  dest 0.160  loot 0.296  R   0.024  eff 0.085  t  133.1s   0.04 ms/act
ppo          L2  win  1.00  dest 0.871  loot 0.938  R   5.793  eff 0.670  t  130.2s  11.25 ms/act
ppo          L3  win  0.62  dest 0.502  loot 0.558  R   2.620  eff 0.296  t  168.0s   8.72 ms/act
ppo          L4  win  0.12  dest 0.268  loot 0.312  R  -0.171  eff 0.149  t  135.2s  10.29 ms/act
ppo          L5  win  0.10  dest 0.184  loot 0.185  R  -0.497  eff 0.098  t  139.8s   8.28 ms/act
