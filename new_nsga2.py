import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from mutation import mutate

class Individual:
    def __init__(self, node):
        self.node = node
        self.scores = node.state.score
        self.rank = None
        self.crowding_distance = 0

class Node():
    def __init__(self, state, n_obj=4, max_child=3, parent=None):
        self.visits = 1
        self.reward = np.zeros(n_obj) 
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

    def perform_reaction(self, mol2):
        mol1 = Chem.MolFromSmiles(self.state.smiles)
        if mol1 is None or mol2 is None:
            return None
            
        reaction_type = random.choice(['suzuki', 'heck', 'ch_activation'])

        if reaction_type == 'suzuki':
            products = suzuki_reaction(mol1, mol2)
        elif reaction_type == 'heck':
            products = heck_reaction(mol1, mol2)
        else:
            products = ch_activation(mol1, mol2)

        return products

    def __repr__(self):
        return "{:s} {:d} [{}]".format(self.state.smiles, self.visits, " ".join(map(str, self.reward)))

def read_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = [line.strip() for line in file if line.strip()]
    return smiles_list

def suzuki_reaction(mol1, mol2):
    reaction = AllChem.ReactionFromSmarts("[C:1]([Br]) + [C:2]([B]) >> [C:1][C:2]")
    return reaction.RunReactants((mol1, mol2))

def heck_reaction(mol1, mol2):
    reaction = AllChem.ReactionFromSmarts("[C:1]([Br]) + [C:2]=[C:3] >> [C:1][C:2][C:3]")
    return reaction.RunReactants((mol1, mol2))

def ch_activation(mol1, mol2):
    reaction = AllChem.ReactionFromSmarts("[C:1]([H]) + [X] >> [C:1][X]")
    return reaction.RunReactants((mol1, mol2))

class NSGA2:
    def __init__(self, population, num_generations=10, crossover_rate=0.9, mutation_rate=0.1):
        self.population = [Individual(node) for node in population]
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def tournament_selection(self, tournament_size=10, num_parents=200):
        selected_parents = []
        for _ in range(num_parents):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda ind: ind.rank)
            selected_parents.append(winner)
        return selected_parents

    def evolve(self):
        for generation in range(self.num_generations):
            print(f"第 {generation + 1} 代")

        # 非优势排序
            fronts = self.fast_non_dominated_sort()

        # 拥挤距离分配
        for front in fronts:
            self.crowding_distance_assignment(front)

        # 选择、交叉和变异
        new_population = []
        parents = self.tournament_selection()  # 使用锦标赛选择
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child_node = self.crossover(parents[i], parents[i + 1])
                new_population.append(Individual(child_node))

                # 执行随机反应
                mol2 = Chem.MolFromSmiles(random.choice(parents))  # 假设 smiles_list 可用
                products = child_node.perform_reaction(mol2)

                if products:
                    for product in products:
                        print("生成的 SMILES:", Chem.MolToSmiles(product[0]))
                        Draw.MolToImage(product[0]).show()

                # **这里是突变的调用**
                mutated_mol = mutate(child_node.state, self.mutation_rate)
                if mutated_mol:
                    child_node.state = mutated_mol  # 更新状态为突变后的分子

        # 用新种群替换旧种群
        self.population = new_population
        return [ind.node for ind in self.population]

