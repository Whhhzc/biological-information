import copy

import bottleneck
import numpy as np

from scores import *

from design_moves import DesignMove
replaceRule = DesignMove("chemblDB3.sqlitdb")


def get_mo_actions(actions, functions, thresholds, t=1.0):
    valid_actions = [] #初始化一个空列表 valid_actions，用于存储筛选出的有效分子动作。
    n_f = len(functions)#计算目标函数的数量
    for s in actions:
        mol = Chem.MolFromSmiles(s)   #使用RDKit库将SMILES字符串转换为分子对象。
        scores = np.zeros(n_f)  #创建一个长度为 n_f 的零数组，用于存储每个目标函数对当前分子的评分。
        for i in range(n_f): #遍历每个目标函数
            scores[i] = functions[i](mol) # 使用每个目标函数对当前分子进行评分，并将结果存储在 scores 数组中。
        if np.all(scores >= t * thresholds):  #检查当前分子的评分是否都大于或等于阈值的 t 倍。这里 t 是一个可选的参数，默认值为1.0。
            valid_actions.append(s)#如果所有评分都满足条件，则将当前分子的SMILES字符串添加到 valid_actions 列表中。
    print("valid actions after constraint {:d}".format(len(valid_actions))) #打印出经过筛选后有效分子动作的数量。
    return valid_actions #返回筛选出的有效分子动作列表。


def get_mo_stage2_actions(current_smiles, actions, functions, thresholds, t=1.0):
    ref_size = Chem.MolFromSmiles(current_smiles).GetNumAtoms() #计算当前分子的原子数量。
    valid_actions = []
    n_f = len(functions)
    for s in actions:  #遍历提供的分子动作列表
        mol = Chem.MolFromSmiles(s)
        mol_size = mol.GetNumAtoms() #计算当前分子的原子数量。
        scores = np.zeros(n_f)
        for i in range(n_f):
            scores[i] = functions[i](mol)
        
        if np.all(scores >= t * thresholds) and mol_size < ref_size:
            valid_actions.append(s)#如果所有评分都满足条件，并且当前分子的大小小于参考分子，
            #则将当前分子的SMILES字符串添加到 valid_actions 列表中
    print("valid actions after constraint {:d}".format(len(valid_actions)))
    #打印出经过筛选后有效分子动作的数量。
    return valid_actions

#是在给定的有效动作中选择得分最高的前 k 个动作
def constraint_top_k(valid_actions, score_func, k=10):
    if len(valid_actions) <= k:  #检查有效动作的数量是否小于或等于 k。
        #如果是这样，那么所有有效动作都是得分最高的，因此直接返回 valid_actions
        return valid_actions
    scores = []
    for s in valid_actions:#遍历每个有效动作
        scores.append(score_func(Chem.MolFromSmiles(s)))
    scores = np.array(scores)
    assert len(scores)==len(valid_actions)

    all_tuple = [(-scores[idx], valid_actions[idx]) for idx in range(len(scores))]
    topk_tuple = sorted(all_tuple)[:k] #对元组列表进行排序，并取前 k 个元素
    topk_actions = [t[1] for t in topk_tuple]  #从排序后的元组中提取出得分最高的 k 个动作。
    return topk_actions  #返回得分最高的 k 个动作

#是从给定的分子状态中获取所有可能的替换动作
def get_actions(state):
    mol = Chem.MolFromSmiles(state)
    if mol is None:
        raise ValueError("Received invalid state: %s" % state)

    valid_actions = set()  #初始化一个空集合 valid_actions，用于存储所有可能的分子修改。
    try:
        print("replace action calculation") #打印一条消息，表示正在计算替换动作
        valid_tmp = _frag_substitution(state, replaceRule)
        print("possible actions: {:d}".format(len(valid_tmp)))  #打印出计算出的替换动作的数量。
        valid_actions.update(valid_tmp) #将计算出的替换动作添加到 valid_actions 集合中
    except:
        pass

    return list(valid_actions)

#从分子中生成可能的替换动作
# def _frag_substitution(smi, rule, min_pairs=1):
#     substitution_actions = rule.one_step_move(query_smi=smi, min_pairs=min_pairs)
#     print('-----------------11111------------------------')
#     #调用 rule 对象的 one_step_move 方法，该方法接受一个SMILES字符串（query_smi）和一个最小配对数（min_pairs）作为参数。one_step_move 方法返回一个可能的替换动作列表，这些动作是基于规则的一步移动。
#     return set(substitution_actions)
#将返回的替换动作列表转换为集合（set），并返回这个集合。集合是一种数据结构，它不允许重复的元素，因此可以确保返回的替换动作是唯一的。

def _frag_substitution(smi, rule, min_pairs=1):
    try:
       substitution_actions = rule.one_step_move(query_smi=smi, min_pairs=min_pairs)
    except Exception as e:
         print(f"An error occurred: {e}")
         
    return set(substitution_actions)

if __name__ == '__main__':
    s = 'C=C(C(C)=C(CCCCCCC)CCCCCCCC)C(CC)=C(C)CCCCCCCCC'
    valid_actions = get_actions(s)
    topk_actions = constraint_top_k(valid_actions, plogp)
   # valid_actions = get_valid_actions(s)改
    #topk_actions = top_k(valid_actions, plogp)改
    # scores = []
    # for a in valid_actions:
    #     scores.append(plogp(Chem.MolFromSmiles(a)))
    # print(sorted(scores)[-10:])
    for a in topk_actions:
        print(a, plogp(Chem.MolFromSmiles(a)))

