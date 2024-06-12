import warnings
warnings.filterwarnings("ignore")
# 导入所需的库和模块
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from graph import subgraphs, train_mask, test_mask
from hcan import HCAN
import torch
from sklearn.metrics import r2_score,mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# 检查是否有可用的CUDA设备，如果有则使用，否则使用CPU

device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置随机数种子以保证代码的可复现性
np.random.seed(seed=5)
torch.manual_seed(seed=5)
torch.cuda.manual_seed_all(seed=5)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# 获取特征和标签
num_feats = subgraphs[0].ndata['feat'].shape[1]
labels = subgraphs[0].ndata['label'].unsqueeze(1).to(torch.float32)
feature = subgraphs[0].ndata['feat']

# 创建模型
model = HCAN(
    num_meta_paths=len(subgraphs),
    in_size=num_feats,
    hidden_size=120,
    out_size=1,
    num_layers=2,
    dropout=0
)

# 清空CUDA缓存
torch.cuda.empty_cache()

# 将特征和图转移到相应的设备上
feature=torch.as_tensor(feature,dtype=torch.float32).to(device)
labels1=torch.as_tensor(labels,dtype=torch.float32).to(device)
g = [graph.to(device) for graph in subgraphs]

# 将模型转移到CUDA设备上
model.cuda()

# 定义损失函数和优化器
loss_fcn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.009, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8)
# 初始化r2分数的列表
r2_1 = []
r2_2 = []
min_loss=10000000000
# 开始训练
for epoch in range(1500):
    model.train()
    logits= model(g, feature)
    loss = loss_fcn(logits[train_mask], labels1[train_mask])

    # 添加L1正则项
    # l1_loss = l1_loss_fcn(logits[train_mask], labels1[train_mask])
    # loss += l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step(loss)
    current_lr = optimizer.param_groups[0]['lr']
    model.eval()
    pred= model(g, feature)
    test_loss = loss_fcn(pred[test_mask], labels1[test_mask])

    logits = logits.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    # labels = labels.cpu().detach().numpy()
    train_r2 = r2_score(labels[train_mask], logits[train_mask])
    r2_1.append(train_r2)
    test_r2 = r2_score(labels[test_mask], pred[test_mask])
    r2_2.append(test_r2)
    if test_loss<min_loss:
        min_loss=test_loss
        a=pred
        mae=mean_absolute_error(labels[test_mask], pred[test_mask])

    print(f'Train Epoch:{epoch} train loss:{loss.item()} r2 score:{train_r2}')
    print(f'Test loss:{test_loss.item()} r2 score: {test_r2}')
#R2图
plt.plot(range(0, len(r2_1) ),r2_1, label='Train r2')
plt.plot(range(0, len(r2_2) ),r2_2, label='test r2')
max_indx=np.argmax(r2_2)
plt.plot(max_indx,r2_2[max_indx],'ks')
show_max='('+str(max_indx)+','+str(r2_2[max_indx])[0:5]+')'
plt.annotate(show_max,xytext=(max_indx,r2_2[max_indx]),xy=(max_indx,r2_2[max_indx]))
print(r2_2[max_indx])
print(min_loss)
print(mae)
plt.xlabel('epochs')
plt.ylabel('R2')# consistent scale
plt.ylim(-0.1,1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.title( "HAGCN R2")
plt.savefig("h.jpg")