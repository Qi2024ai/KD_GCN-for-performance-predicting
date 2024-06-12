import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# 读取csv文件
data = pd.read_csv('medium C steel_ori.csv',encoding='gbk')

# 假设前12列是特征，第13列是目标变量
X = data.iloc[:, 1:11].values
y = data.iloc[:, 12].values

# 对输入进行归一化处理
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# # 将预处理后的输入保存为新的csv文件
# pd.DataFrame(X).to_csv('data_Gyh.csv', index=False)

# 划分训练集和测试集
X_train=X[0:112]
X_test=X[112:140]
y_train=y[0:112]
y_test=y[112:140]
# = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练逻辑回归模型
model = LinearRegression()  
model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 计算并打印评价指标

print(f'Accuracy: {r2_score(y_test, y_pred)}')
# print(f'Precision: {precision_score(y_test, y_pred, average="macro", zero_division=1)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
# print(f'F1 Score: {f1_score(y_test, y_pred, average="macro")}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
# print(f'PR AUC Score: {average_precision_score(y_test, y_pred)}')
