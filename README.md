# Benchmark Test (gsk3, jnk3, qed, sa)

## STAGE 1
As the number of starting molecules provided in "./libs/start_mols/start_mols_task1.csv" is 96, an user have to execute the "frag_mcts_mo_stage1.py" 96 times.

The following commands were written with the optimal hyperparameters presented in the paper.
```
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 0 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 1 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 2 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 3 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 4 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 5 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 6 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 7 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 8 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 9 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 10 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 11 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 12 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 13 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 14 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 15 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 16 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 17 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 18 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 19 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 20 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 21 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 22 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 23 --seed 0 --scalar 0.7

python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 24 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 25 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 26 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 27 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 28 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 29 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 30 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 31 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 32 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 33 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 34 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 35 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 36 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 37 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 38 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 39 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 40 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 41 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 42 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 43 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 44 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 45 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 46 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 47 --seed 0 --scalar 0.7

python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 48 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 49 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 50 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 51 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 52 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 53 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 54 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 55 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 56 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 57 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 58 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 59 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 60 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 61 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 62 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 63 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 64 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 65 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 66 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 67 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 68 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 69 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 70 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 71 --seed 0 --scalar 0.7

python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 72 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 73 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 74 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 75 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 76 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 77 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 78 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 79 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 80 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 81 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 82 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 83 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 84 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 85 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 86 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 87 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 88 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 89 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 90 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 91 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 92 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 93 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 94 --seed 0 --scalar 0.7
python frag_mcts_mo_stage1.py --goal gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --mol_idx 95 --seed 0 --scalar 0.7
```

## STAGE 2
For each starting molecule, the outputs of stage 1 are stored in "./results_visualization/gsk3b_jnk3_qed_sa_stage1/task1/" with the following filenames:
- "gsk3b_jnk3_qed_sa_maxchild_7_sim_10_scalar_0.7_idx_{idx}_seed_0.txt"
- "gsk3b_jnk3_qed_sa_maxchild_7_sim_10_scalar_0.7_idx_{idx}_seed_0.pkl"

To execute the stage 2 script, at first, an user have to integrate all outputs into a single .csv file and save it with the below filepath:
- "/libs/gsk3b_jnk3_qed_sa_stage1/result_start_mols_task1_seed_0.csv"

After integrating the stage 1 outputs, run the stage 2 script with the following commands:
```
python frag_mcts_mo_stage2.py --goal gsk3b_jnk3_qed_sa --constraint gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --scalar 0.7 --start_idx 0 --end_idx 143 --seed 0 
python frag_mcts_mo_stage2.py --goal gsk3b_jnk3_qed_sa --constraint gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --scalar 0.7 --start_idx 143 --end_idx 286 --seed 0 
python frag_mcts_mo_stage2.py --goal gsk3b_jnk3_qed_sa --constraint gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --scalar 0.7 --start_idx 286 --end_idx 429 --seed 0 
python frag_mcts_mo_stage2.py --goal gsk3b_jnk3_qed_sa --constraint gsk3b_jnk3_qed_sa --start_mols task1 --max_child 7 --num_sims 10 --scalar 0.7 --start_idx 429 --end_idx 570 --seed 0 
```
