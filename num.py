import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch.nn.functional as F

# 读取CSV文件
df = pd.read_csv('medium C steel_ori.csv')

# 创建一个归一化对象
scaler = MinMaxScaler()

# 定义需要归一化的列数
N = 11  # 根据需求修改这个值

# 对前N列进行归一化
for col in df.columns[1:N]:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
num=df.values[:,1:N]
# num
num=torch.tensor(num)
# semantic_num值
df2 = pd.read_csv('TransE_240.csv',header=None)
semantic= torch.tensor(df2.values.tolist())

# semantic=torch.cat([semantic, semantic, semantic], dim=0)
scale = torch.floor_divide(torch.arange(semantic.shape[1]) ,torch.floor_divide(semantic.shape[1] , 10))
semantic *= num[:, scale]
features=semantic