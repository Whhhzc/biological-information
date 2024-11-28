import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--goal', type=str, default='gsk3b_jnk3')
parser.add_argument('--start_mols', type=str, default='task1')
parser.add_argument('--max_child', type=int, default=5)
parser.add_argument('--num_sims', type=int, default=20)
parser.add_argument('--scalar', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# 读取CSV文件并将数值列转换为浮点型
def read_csv(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=0,usecols=[0,1,2,3,4,5,8], 
                       names=["size", "smiles", "gsk3b", "jnk3", "qed", "sa", "max_or_val"])
    
    # 将数值列转换为浮点型
    data["gsk3b"] = pd.to_numeric(data["gsk3b"], errors='coerce')
    data["jnk3"] = pd.to_numeric(data["jnk3"], errors='coerce')
    data["qed"] = pd.to_numeric(data["qed"], errors='coerce')
    data["sa"] = pd.to_numeric(data["sa"], errors='coerce')

    # 过滤掉无效的 SMILES 字符串
    data = data[data['smiles'].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
    
    return data

# 计算成功率 (SR)
def calculate_sr(data):
    success = data[(data["qed"] >= 0.5) | (data["sa"] >= 0.5) |
                   (data["gsk3b"] >= 0.5) |(data["jnk3"] >= 0.5)]
    sr = len(success) / len(data)
    return sr

# 计算新颖性 (Nov)
def calculate_nov(data, reference_fps):
    novel_count = 0
    for _, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3)
            max_similarity = max([DataStructs.FingerprintSimilarity(fp, ref_fp) for ref_fp in reference_fps])
            if max_similarity < 0.5:
                novel_count += 1
    nov = novel_count / len(data)
    return nov

# 计算多样性 (Div)
def calculate_div(data):
    fps = []
    for _, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3))
    
    n = len(fps)
    if n < 2:
        return 0.0

    sum_div = 0
    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            sum_div += (1 - similarity)
    
    div = (2 / (n * (n - 1))) * sum_div
    return div

# 运行计算
def main(file_path, reference_smiles_list):
    # 读取数据
    data = read_csv(file_path)
    
    # 计算参考指纹
    reference_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2) 
                     for smiles in reference_smiles_list]

    # 计算SR、Nov、Div
    sr = calculate_sr(data)
    nov = calculate_nov(data, reference_fps)
    div = calculate_div(data)

    print(f"SR: {sr:.4f}, Nov: {nov:.4f}, Div: {div:.4f}")

start_mols_fn = 'libs/start_mols/start_mols_' + args.start_mols + '.csv'
start_mols_df = pd.read_csv(start_mols_fn)
start_mol_list = start_mols_df['smiles'].tolist()
save_dir_base = 'NSGA2'
#save_dir_base = 'results_visulization/' + args.goal + '_stage1/' + args.start_mols
output_file = os.path.join(save_dir_base, f"{args.goal}.csv")
#output_file = os.path.join(save_dir_base, f"{args.goal}_maxchild_{args.max_child}_sim_{args.num_sims}_scalar_{args.scalar}_seed_{args.seed}_results.txt")
reference_smiles_list = start_mol_list
results_file_path = output_file  

main(results_file_path, reference_smiles_list)
