import pandas as pd
import dgl
import torch
from collections import defaultdict
from num import features
# 读取csv文件
df = pd.read_csv('medium C steel_ori.csv')

# 创建一个字典来存储每个标签的节点
label_dict1 = defaultdict(list)
label_dict2 = defaultdict(list)
label_dict3 = defaultdict(list)
for i, row in df.iterrows():
    label_dict1[row['label1']].append(row['id'])
    label_dict2[row['label2']].append(row['id'])
    label_dict3[row['label3']].append(row['id'])

# 创建边的列表
edges1 = []
edges2 = []
edges3 = []
for nodes in label_dict1.values():
    # 对于每个标签，将所有节点两两连接
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            edges1.append((nodes[i], nodes[j]))
for nodes in label_dict2.values():
    # 对于每个标签，将所有节点两两连接
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            edges2.append((nodes[i], nodes[j]))
for nodes in label_dict3.values():
    # 对于每个标签，将所有节点两两连接
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            edges3.append((nodes[i], nodes[j]))

# 创建图
g = dgl.heterograph({
    ('node', 'edge_type1', 'node'): edges1,
    ('node', 'edge_type2', 'node'): edges2,
    ('node', 'edge_type3', 'node'): edges3
})
g = dgl.to_bidirected(g)
# 创建一个空列表来存储同构图
subgraphs = []

# 遍历图的每一种边类型
for etype in g.etypes:
    # 为当前的边类型创建一个子图
    subgraph = g.edge_type_subgraph([etype])
    # 将子图添加到列表中
    subgraphs.append(subgraph)

g.ndata['feat'] = features

# 将标签添加到图中
# labels1 = torch.tensor(df['YS'].values)
# labels1_1 = torch.tensor(df['UTS'].values)
labels1_2 = torch.tensor(df['YS'].values)
# a=torch.stack((labels1,labels1_1,labels1_2),dim=1)
# g.ndata['label'] = torch.stack((labels1,labels1_1,labels1_2),dim=1)
g.ndata['label'] =labels1_2
# 创建一个值为True的Tensor，长度为112
true_tensor = torch.ones(112, dtype=torch.bool)
# 创建一个值为False的Tensor，长度为28
false_tensor = torch.zeros(28, dtype=torch.bool)

# zero_tensor=torch.zeros(280, dtype=torch.bool)
# 将两个Tensor沿着第0维（行）拼接起来
train_mask = torch.cat((true_tensor,false_tensor))
# 创建一个值为True的Tensor，长度为112
false_tensor = torch.zeros(112, dtype=torch.bool)
# 创建一个值为False的Tensor，长度为28
true_tensor = torch.ones(28, dtype=torch.bool)
# zero_tensor=torch.zeros(280, dtype=torch.bool)
# 将两个Tensor沿着第0维（行）拼接起来
test_mask = torch.cat((false_tensor,true_tensor))

if __name__ =='__main__':
    # 打印同构图
    for i, subgraph in enumerate(subgraphs):
        print(f'Subgraph {i}:')
        print(subgraph)
