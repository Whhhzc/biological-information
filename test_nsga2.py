import numpy as np
import random
import pandas as pd
from rdkit import Chem
import os
from actions import get_actions, get_mo_actions
from scores import *

import numpy as np
import math

from rdkit import Chem

import random

import hashlib
import argparse
import time, os, datetime
import pickle as pkl

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--goal', type=str, default='gsk3b_jnk3')
parser.add_argument('--start_mols', type=str, default='task1')
parser.add_argument('--max_child', type=int, default=5)
parser.add_argument('--num_sims', type=int, default=20)
parser.add_argument('--scalar', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=0)
def get_score_function(name):
    if name == 'plogp':
        sf = plogp
    elif name == 'qed':
        sf = qed
    elif name == 'esol':
        sf = esol
    elif name == 'sa':
        sf = sa
    elif name == 'rges':
        sf = rges
    elif name == 'gsk3b':
        sf = gsk3b
    elif name == 'jnk3':
        sf = jnk3
    else:
        print("invalid goal!")
    return sf
args = parser.parse_args()
functions = []
goals = args.goal.split("_")
for g in goals:
    functions.append(get_score_function(g))
print('--------------------------',goals)
n_obj=len(goals)
class Individual:
    def __init__(self, node):
        self.node = node
        self.scores = node.state.score
        self.rank = None
        self.crowding_distance = 0

class Node:
    def __init__(self, state, n_obj=n_obj, max_child=3, parent=None):
        self.visits = 1
        self.reward = np.zeros(n_obj)  # float?
        self.state = state
        self.parent = parent
        self.max_child = max_child
        self.children = []

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)

    def update(self, r):
        self.reward += r
        self.visits += 1

    def fully_expanded(self):
        return len(self.children) == self.max_child

    def __repr__(self):
        s = "{:s} {:d} ".format(self.state.smiles, self.visits)
        tmp = "[ "
        for i in range(self.reward.shape[0]):
            tmp += str(self.reward[i]) + " "
        tmp += "]"
        s += tmp
        return s

class NSGA2:
    def __init__(self, population, num_generations=10, crossover_rate=0.9, mutation_rate=0.3):
        
        self.population = [Individual(node) for node in population]
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def fast_non_dominated_sort(self):
        fronts = [[]]
        for p in self.population:
            p.dominated_solutions = []
            p.domination_count = 0
            for q in self.population:
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts

    def crowding_distance_assignment(self, front):
        if len(front) == 0:
            return

        num_objectives = len(front[0].scores)
        for p in front:
            p.crowding_distance = 0

        for i in range(num_objectives):
            front.sort(key=lambda x: x.scores[i])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (front[j + 1].scores[i] - front[j - 1].scores[i])

    def dominates(self, individual1, individual2):
        return all(x >= y for x, y in zip(individual1.scores, individual2.scores)) and any(x > y for x, y in zip(individual1.scores, individual2.scores))

    def selection(self):
        parents = []
      
        if len(self.population) < 2:
            parents = self.population * 2
        else:
            for _ in range(len(self.population)):
                p1, p2 = random.sample(self.population, 2)
                print('p1p1p1p1p1',p1.node)
                print('p2p1p1p1p1',p2.node)
                if p1.rank < p2.rank or (p1.rank == p2.rank and p1.crowding_distance > p2.crowding_distance):
                    parents.append(p1)
                else:
                    parents.append(p2)
        return parents

    def crossover(self, parent1, parent2):
        print('parent1parent1parent1',type(parent1.node))
        print('parent2parent1parent1',parent2.node)
        if random.random() < self.crossover_rate:
            return self.mutate(self.crossover_func(parent1.node, parent2.node))
            
        else:
            return parent1.node

    def crossover_func(self, node1, node2):
        # 获取两个父节点的 SMILES 字符串
        smiles1 = node1.state.smiles
        smiles2 = node2.state.smiles

        # 随机选择一个交叉点
        crossover_point = random.randint(1, min(len(smiles1), len(smiles2)) - 1)

        # 创建新的 SMILES 字符串
        new_smiles = smiles1[:crossover_point] + smiles2[crossover_point:]
        print('smiles1smiles1smiles1',smiles1)
        print('smiles2smiles2smiles2',smiles2)
        print('new_smilesnew_smilesnew_smiles',new_smiles)
        return Node(State(new_smiles))


    def mutate(self, node):
        if random.random() < self.mutation_rate:
        # 尝试多次生成新状态，直到得到一个与原状态不同的有效新状态
            max_attempts = 200
            for _ in range(max_attempts):
                new_state = node.state.next_state()
                if new_state.smiles!= node.state.smiles:
                    return Node(new_state)
            print(f"Failed to generate a valid different state after {max_attempts} attempts.")
            return node
        return node

    def is_valid_state(self, state):
    # 这里添加对新状态的有效性检查，例如检查分子结构的合理性等
        mol = Chem.MolFromSmiles(state.smiles)
        if mol is None:
            return False
        return True

    def evolve(self):
        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}")

            fronts = self.fast_non_dominated_sort()

            for front in fronts:
                self.crowding_distance_assignment(front)

            new_population = []
            parents = self.selection()
            
            # print('parentsparentsparents11',parents)
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child_node = self.crossover(parents[i], parents[i + 1])
                    new_population.append(Individual(child_node))
                    print('child_nodechild_nodechild_node',child_node)
            self.population = new_population
        return [ind.node for ind in self.population]


#读取分子  
start_mols_fn = 'libs/start_mols/start_mols_task1.csv'
start_mols_df = pd.read_csv(start_mols_fn)
start_mol_list = start_mols_df['smiles'].tolist()

n_obj = len(goals)
MAX_LEVEL = 5

class State():

    def __init__(self, smiles='', level=0):
        self.smiles = smiles
        self.level = level
        self.score = []
        if self.smiles:
            print("smiles score ")
            mol = Chem.MolFromSmiles(self.smiles)
            for f in functions:
                self.score.append(f(mol))
        else:
            print("smiels score else")
            self.score = [0.0] * n_obj

        self.valid_actions = self.get_valid_actions()

    def get_valid_actions(self):
        print('---------111---------')
        actions = get_actions(state=self.smiles)
        print('--------222----------')
        mo_actions = get_mo_actions(actions, functions=functions, thresholds=np.array(self.score))
        return mo_actions #constraint_top_k(valid_actions_constraint, score_func=target_function, k=args.top_k)

    def next_state(self):
        if len(self.valid_actions) == 0:
            self.level = MAX_LEVEL
            return self
        #print(self.valid_actions)
        s = random.choice(self.valid_actions)
        next = State(s, self.level + 1)
        return next

    def terminal(self):
        return self.level == MAX_LEVEL
    
    def dominate_score(self, v):
        d = len(self.score)
        r = [1.0] * d
        for i in range(d):
            if self.score[i] < v[i]:
                r[i] = 0.0
        return np.array(r)

    def reward(self):
        global max_score
        global count
        global val_score
        count += 1

        if np.all(np.array(self.score) > 0.5):
            if self.smiles not in val_score:
                val_score[self.smiles] = self.score

        # max_score dictionary: key smiles, value [r1, r2,..]
        total_reward = np.zeros(len(self.score))
        n = len(max_score)
        keys_to_delete = []
        new_max_score_found = False
        always_uncomparable = True
        for s, r in max_score.items():
            win = self.dominate_score(r)
            total_reward += win
            if sum(win) == float(len(self.score)):
                keys_to_delete.append(s)
                new_max_score_found = True
                always_uncomparable = False
            elif sum(win) == 0.0:
                always_uncomparable = False

        for k in keys_to_delete:
            max_score.pop(k, None)
        if new_max_score_found or always_uncomparable:
            if self.smiles not in max_score:
                max_score[self.smiles] = self.score
        return total_reward/n


    def __hash__(self):
        # override default built-in function to compare whether two objects are the same
        return int(hashlib.md5(str(self.smiles).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        s = "level {:d} state {:s} score ".format(self.level, self.smiles)
        tmp = "[ "
        for r in self.score:
            tmp += str(r)
            tmp += " "
        tmp += "]"
        s += tmp
        return s



def UCTSEARCH(budget, root):
    # n_budget simulations
    # for each simulation, reach a node, do sth, backprop one time
    # at last, search for best child and return

    # NSGA2
    population = [root]
    nsga2_optimizer = NSGA2(population)
    
    for iter in range(budget):
        print("===== sim_iter {:d} ======".format(iter))
       

       

        # Apply NSGA-II to optimize the population
        population = nsga2_optimizer.evolve()
        
        # Select the next root node from the evolved population
        root = random.choice(population)
        
        print("new root: ", root.state.smiles, root.reward, root.visits)
        
        for k, v in max_score.items():
            print("max score: ", v, k, Chem.MolFromSmiles(k).GetNumAtoms())
        for k, v in val_score.items():
            print("val score: ", v, k, Chem.MolFromSmiles(k).GetNumAtoms())
        
    return root # BESTCHILD(root, 0) # exploration_scalar=0




def save_results(output_file, max_score, val_score):
    with open(output_file, "a") as f:
        for s, r in max_score.items():
            print('max_scoremax_scoremax_score',max_score)
            l = str(Chem.MolFromSmiles(s).GetNumAtoms()) + ' ' + s + ' '
            for x in r:
                l += str(x) + ' '
            l += str(qed(Chem.MolFromSmiles(s))) + ' '
            l += str(sa(Chem.MolFromSmiles(s))) + ' '
            l += str(1)
            l += '\n'
            f.write(l)

        for s, r in val_score.items():
            l = str(Chem.MolFromSmiles(s).GetNumAtoms()) + ' ' + s + ' '
            for x in r:
                l += str(x) + ' '
            l += str(qed(Chem.MolFromSmiles(s))) + ' '
            l += str(sa(Chem.MolFromSmiles(s))) + ' '
            l += str(0)
            l += '\n'
            f.write(l)



def run_single_molecule(smiles, mol_idx, output_file):
    print(f"Start processing molecule {mol_idx}: {smiles}")
    
    global max_score, val_score, count
    max_score = {}
    val_score = {}
    count = 0
    
    root = Node(State(smiles))
   
    population = [root]
    nsga2_optimizer = NSGA2(population)
    population = nsga2_optimizer.evolve()
    root = random.choice(population)
    
    new_pop=str(population)
    print('new_popnew_popnew_pop',new_pop[0],type(new_pop))
    space_index = new_pop.find(' ')
    print('smilessmilessmiles',smiles,type(smiles))
    # 如果找到了空格，则返回空格之前的部分作为SMILES字符串
    if space_index != -1:
        smiles = new_pop[:space_index]
    else:
        # 如果没有找到空格，则整个字符串可能就是SMILES字符串（或者格式不正确）
        smiles = new_pop
    smiles=smiles[1:]
    print('smilessmilessmiles',smiles,type(smiles))
    print('child_nodechild_nodechild_node222',population[0])
    max_score[smiles] = np.array(root.state.score)

    save_results(output_file, max_score, val_score)



# output_file= 'test' + str(args.goal) + '.csv'
strgoal='_'.join(goals)
output_file = os.path.join('NSGA2/', f"{strgoal}.csv")


if __name__ == "__main__":

  for mol_idx, smiles in enumerate(start_mol_list):
      run_single_molecule(smiles, mol_idx, output_file)