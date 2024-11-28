import subprocess
for mol_idx in range(1, 98):
    command = f"python frag_mcts_mo_stage1.py --goal gsk3b --start_mols task1 --max_child 5 --num_sims 20 --mol_idx {mol_idx} --seed 0 --scalar 0.7"
    subprocess.call(command, shell=True)