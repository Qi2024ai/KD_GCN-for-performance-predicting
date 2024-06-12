import torch.nn as nn
import torch
import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from math import sqrt
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"
# # 设置随机数种子以保证代码的可复现性
np.random.seed(seed=5)
torch.manual_seed(seed=5)
torch.cuda.manual_seed_all(seed=5)

class MLP(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.n_hidden_1=n_hidden_1
        self.n_hidden_2=n_hidden_2
        super(MLP,self).__init__()
        # self.q = nn.Linear(self.n_hidden_2,self.n_hidden_2)
        # self.k = nn.Linear(self.n_hidden_2,self.n_hidden_2)
        # self.v = nn.Linear(self.n_hidden_2,self.n_hidden_2)
        # self._norm_fact = 1 / sqrt(self.n_hidden_2)
        self.layer1 = nn.Sequential(nn.Linear(self.in_dim,self.n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(self.n_hidden_1, self.n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(self.n_hidden_2, self.out_dim))
    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        # Q = self.q(x) # Q: batch_size * seq_len * dim_k
        # K = self.k(x) # K: batch_size * seq_len * dim_k
        # V = self.v(x)
        # atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        # output = torch.bmm(atten,V)
        x = self.layer3(x)
        return x

dt=pd.read_csv('medium C steel_ori.csv')
dataset=dt.values
X=dataset[:,1:11].astype(float)
dt=pd.read_csv('medium C steel_ori.csv')
dataset=dt.values
Y=dataset[:, -6].astype(float)
scaler = MinMaxScaler()  # 实例化
X = scaler.fit_transform(X)  # 标准化特征
Y = scaler.fit_transform(Y.reshape(-1,1))  # 标准化标签
Y=Y.reshape(-1,1)
# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前
X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
Y = torch.tensor(Y, dtype=torch.float32)
train=torch.utils.data.TensorDataset(X[0:112],Y[0:112])
validation_test=torch.utils.data.TensorDataset(X[112:140],Y[112:140])
# test, validation = torch.utils.data.random_split(validation_test,[14,14]) 
train_data = torch.utils.data.DataLoader(train,shuffle=True,batch_size=64)

Model = MLP(in_dim=10,n_hidden_1=46,n_hidden_2=45,out_dim=1).to(device)
optimizer = optim.Adam(Model.parameters(), lr=0.0007,weight_decay=1e-4) 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.8) # 使用Adam算法更新参数
loss_fn= nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
epochs=2500
r2_1=[]
r2_2=[]
min_loss=10000000
for epoch in range(epochs):
    if epoch>800 and epoch<1600:
        for p in optimizer.param_groups:
            p['lr']=0.0006
    if epoch>1600 and epoch<2500:
        for p in optimizer.param_groups:
            p['lr']=0.0005
    Model.train()
    logit=[]
    targets=[]
    for data, target in train_data:
        data, target = data.to(device), target.to(device)
        # Compute prediction error
        logits = Model(data)
        loss = loss_fn(logits, target)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
        logits=logits.cpu().detach().numpy()
        target=target.cpu().detach().numpy()
        logit.extend(logits)
        targets.extend(target)
    train_r2=str(r2_score(targets, logit))
    r2_1.append(float(train_r2))
    print('Train Epoch:{} train loss:{} r2 score:{}'.format(epoch,loss.item(),train_r2))
    Model.eval()
    test_y=[]
    pred=[]

    for test_x, test_ys in validation_test:
        test_x, test_ys = test_x.to(device), test_ys.to(device)
        predictions = Model(test_x)
        predictions=predictions.cpu().detach().numpy()
        pred.append(predictions[0])
        test_y.append(test_ys[0])
    test_y=torch.tensor(test_y,device ='cpu')
    Trtest_loss = loss_fn(torch.tensor(pred), test_y )
    pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1))  # 将数据恢复至归一化之前
    test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))
    # 均方误差计算
    test_loss = loss_fn(torch.tensor(pred ,dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
    if test_loss<min_loss:
        min_loss=test_loss
        a=pred
        mae=mean_absolute_error(test_y, pred)
    # np.savetxt("pred.txt",a)
    test_r2=str(r2_score(test_y, pred))
    r2_2.append(float(test_r2))
    print('测试集均方误差：'+str(test_loss.detach().numpy())+"\t"+"r2 score:"+test_r2)
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
plt.title( "MLP R2")
# plt.show()
# a = sys.argv[1]
# plt.savefig("./num/{}_num.jpg".format(1))