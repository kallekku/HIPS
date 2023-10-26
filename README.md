Hierarchical Imitation Planning with Search
======
Code for the papers  
- *Hierarchical Imitation Learning with Vector Quantized Models* (ICML 2023 poster)  
- *Hybrid Search for Efficient Planning with Completeness Guarantees* (NeurIPS 2023 poster)  

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
1) Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1h4wU48vO9PX-RLzwfzV6KOBJbKLNLYPD) 
2) Perform the following steps (in parallel)
   1) Train the HIPS detector and the subgoal conditioned low-level policy with `reinforce.py`
   2) Train the continuous vqvae with `vqvae.py` using the argument `--continuous`
   3) Train the dynamics model with `train_model.py` (optional, you can use the env dynamics as an alternative) 
   4) Create the distance dataset with `distance_dataset_creator.py` and train the distance function with `train_distance_function.py`
4) Create the discrete VQVAE dataset with `vqvae_dataset_creator.py`
5) Train the discrete vqvae with `vqvae.py`
6) Create a dataset for training the prior and low-level BC policy with `prior_dataset_creator.py`
7) Train the prior and low-level BC policy with `prior.py`

Evaluation
------------
The main script for evaluation is search.py. The command to use is:
```
python search.py --env <ENV> --policy <POLICY> --vqvae <VQVAE> --prior <PRIOR> --heuristic <DIST_FUNC> --jobs <N> \
  --epsilon <E> --hybrid [--K <K>] [--ada] [--gbfs] [--astar] [--step\_cost] [--model <MODEL>]
```

For evaluating
- with $\varepsilon \to 0$, use `--ada`
- baseline HIPS (no hybrid search), do not use `--hybrid`
- with GBFS or A*, use `--gbfs` or `--astar` respectively. Then, a prior is not needed, but the value of `K` must be specified. When A* is used, you should also use `--step_cost`
- with model, include `--model <MODEL>`
