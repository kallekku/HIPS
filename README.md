Hybrid Search for Efficient Planning with Completeness Guarantees
======
Code for the paper

Environment
------------
Execute the following commands to create the environment:
```
conda create -n hips python=3.9
conda activate hips
pip3 install -r requirements.txt
```

Training
------------
Perform the following steps to train the models. Steps 2a-2d can be done in parallel. Steps 3-6 depend on 2a and 2b.
1) Download the datasets from: https://drive.google.com/drive/folders/1h4wU48vO9PX-RLzwfzV6KOBJbKLNLYPD
2) Perform the following steps (in parallel)
2a) Train the HIPS detector and the subgoal conditioned low-level policy with reinforce.py
2b) Train the continuous vqvae with vqvae.py using the argument --continuous
2c) Train the dynamics model with train_model.py (optional, you can use the env dynamics as an alternative) 
2d.1) Create the distance dataset with distance_dataset_creator.py 
2d.2) Train the distance function with train_distance_function.py
3) Create the discrete VQVAE dataset with vqvae_dataset_creator.py
4) Train the discrete vqvae with vqvae.py
5) Create a dataset for training the prior and low-level BC policy with prior_dataset_creator.py
6) Train the prior and low-level BC policy with prior.py

Evaluation
------------
The main script for evaluation is search.py. The command to use is:
```
python search.py --env <ENV> --policy <POLICY> --vqvae <VQVAE> --prior <PRIOR> --heuristic <DIST_FUNC> --jobs <N> --epsilon <E> --K <K> --hybrid
```

For evaluating
- with $\varepsilon \to 0$, use option --ada
- baseline HIPS (no hybrid search), do not use --hybrid
- with GBFS or Astar (Appendix D), use --gbfs or --astar respectively. Then a prior is not needed.
- with Astar is used, you should also use --step\_cost
- with model, include --model <MODEL>
