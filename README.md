# drl4dynamicrouting
Reinforcement Learning Final Project Code

## Requirements:

* Python 3.7
* pytorch 1.3.1
* matplotlib
* pandas

## TSP
For dynamic TSP, nearest neighbor is used as the baseline. To run the benchmark, go to '''dynamic_tsp/dynamic_nn''' and run the command '''python dynalocTSP.py'''

The DRL code for dynamic TSP is in '''dynamic_tsp/drl'''
To train the model, run '''python trainer.py'''
To run the benchmark using the provided checkpoint for {num_nodes} number of nodes, run '''python trainer.py --checkpoint tsp/20/12_46_22.667847/ --test --batch_size 1 ---nodes {num_nodes}'''

## VRP
For dynamic VRP, a variation of the Clarke-Wright algorithm is used as the baseline. This variation generates routes like normal, but immediately returns to the depot if the current route would violate demand constraints due to the dynamic demand. The algorithm is rerun (with visited nodes ignored) each time this happens. To run the benchmark, go to '''dynamic_drp/dynamic_cw''' and run the command python dynademandVRP.py

The DRL code for dynamic VRP is in '''dynamic_vrp/drl'''
To train the model, run '''python trainer.py --task vrp'''
To run the benchmark using the provided checkpoint for {num_nodes} number of nodes, run '''python trainer.py --checkpoint vrp/20/22_55_44.077690/ --test --batch_size 1 ---nodes {num_nodes}'''
