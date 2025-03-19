import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class GSK3BCalculator():

    def __init__(self, **kwargs):# 初始化方法
        super(GSK3BCalculator, self).__init__(**kwargs)# 调用父类的初始化方法
        self.model = self._load_model()# 调用_load_model方法加载预训练模型
    def _load_model(self):
        # 打开预训练模型文件
        with open('/home/zcchong/jupyterlab/MolSearch-main/MCTS/score_modules/GSK3B_Score/gsk3b.pkl', 'rb') as f:	#score_modules/GSK3B_Score/
            model =  pickle.load(f, encoding='iso-8859-1')
        return model
    
    def _get_morgan_fingerprint(self, mol, radius=3, nBits=1024):
        # 使用RDKit计算分子的Morgan指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False)
        fp_bits = fp.ToBitString() # 将指纹向量转换为bit字符串
        #finger_print = np.fromstring(fp_bits, 'u1')-ord('0')
        finger_print = np.array(list(map(int, fp_bits))) # 将bit字符串转换为numpy数组
        return finger_print# 返回最终的指纹向量

    def get_score(self, mol):
        fp = self._get_morgan_fingerprint(mol).astype(np.float32).reshape(1, -1)# 计算分子的Morgan指纹,并将其转换为float32类型和(1, 1024)的形状
        score = self.model.predict_proba(fp)[0, 1] # 使用预训练模型预测分子的GSK3B活性概率
        return score


if __name__ == '__main__':
    s = 'CNC(=O)c1c(F)cccc1Nc1nc(Nc2cc3c(cc2OC)CCN3C(=O)CN(C)C)nc2[nH]ccc12'
    mol = Chem.MolFromSmiles(s)
    gsk3b_calculator=GSK3BCalculator()
    print(gsk3b_calculator.get_score(mol))