from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit.Chem import QED
from score_modules.ESOL_Score.esol import ESOLCalculator
from score_modules.SA_Score import sascorer
#from score_modules.RGES_Score.rges import RGESCalculator
from score_modules.GSK3B_Score.gsk3b import GSK3BCalculator
from score_modules.JNK3_Score.jnk3 import JNK3Calculator
#from score_modules.COVID_Score.covid import COVIDCalculator


esol_calculator = ESOLCalculator()
#rges_calculator = RGESCalculator()
gsk3b_calculator = GSK3BCalculator()
jnk3_calculator = JNK3Calculator()
#covid_calculator = COVIDCalculator()

#计算给定分子的 GSK3B 得分
def gsk3b(mol):
    return gsk3b_calculator.get_score(mol)

# 计算给定分子的 JNK3 得分 
def jnk3(mol):
    return jnk3_calculator.get_score(mol)

#计算给定分子的"药物相似性定量估计"(QED)得分
def qed(mol):
    return QED.qed(mol)

#计算给定分子的合成可访问性(Synthetic Accessibility, SA)得分
def sa(mol):
    sa_score = sascorer.calculateScore(mol)
    normalized_sa = (10-sa_score) / 9
    return normalized_sa


#def npc1(mol):
#    return covid_calculator.get_score(mol, 'npc1')


#def insig1(mol):
#    return covid_calculator.get_score(mol, 'insig1')


#def hmgcs1(mol):
#    return covid_calculator.get_score(mol, 'hmgcs1')


#计算给定分子的溶解度(ESOL)得分.
def esol(mol):
    return esol_calculator.calc_esol(mol)


# def rges(mol):
#     return -1 * rges_calculator.rges_score(mol)


def get_largest_ring_size(mol):
    # 获取分子中所有环的信息,存储在 cycle_list 列表中
    cycle_list = mol.GetRingInfo().AtomRings()
    # 如果存在环,则找到最大环的大小
    # 如果没有环,则环大小为 0
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def plogp(mol):
     # 计算分子的 logP 值
    log_p = Descriptors.MolLogP(mol)
    # 计算分子的合成可访问性(SA)评分
    sas_score = sascorer.calculateScore(mol)
    # 获取分子中最大环的大小
    largest_ring_size = get_largest_ring_size(mol)
    # 计算基于最大环大小的"环分数"
    cycle_score = max(largest_ring_size - 6, 0)
    #计算 p-logP 值,方法是减去 SA 评分和环分数
    p_logp = log_p - sas_score - cycle_score
    return p_logp


def qed_sa(mol):
    # 计算分子的 QED 得分
    qed_score = qed(mol)
    # 计算分子的 SA 得分
    sa_score = sa(mol)
    # 将 SA 得分归一化到 0-1 范围内
    nomalized_sa = (sa_score-1)/9.0
    # 返回 QED 得分加上归一化的 SA 得分
    return qed_score+sa_score


if __name__=='__main__':
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    import pandas as pd

    df = pd.read_csv('libs/drug_like_cluster_info.csv')
    smiles = df['smiles'].tolist()
    # 初始化用于存储计算结果的列表
    plogp_scores = []
    qed_scores = []
    sa_scores = []
    rges_scores = []
    esol_scores = []
    gsk3b_scores = []
    jnk3_scores = []
    # 遍历每个 SMILES 字符串,计算相应分子的各项指标
    i = 0
    for s in smiles:
        if i%100 == 0:
            print(i)
        mol = Chem.MolFromSmiles(s)
        plogp_scores.append(plogp(mol))
        qed_scores.append(qed(mol))
        sa_scores.append(sa(mol))
        # rges_scores.append(rges(mol))
        esol_scores.append(esol(mol))
        gsk3b_scores.append(gsk3b(mol))
        jnk3_scores.append(jnk3(mol))
        i+=1
    # 将计算结果添加到原始 DataFrame 中
    df['plogp'] = plogp_scores
    df['qed'] = qed_scores
    df['sa'] = sa_scores
    df['gsk3b'] = gsk3b_scores
    df['jnk3'] = jnk3_scores
    df['rges'] = rges_scores
    df['esol'] = esol_scores
    # 将结果保存为新的 CSV 文件
    df.to_csv('libs/drug_like_cluster_prop_info.csv', index=False)
