import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import AdaBoostRegressor  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
  
# 读取csv文件
data = pd.read_csv('medium C steel_ori.csv',encoding='gbk')

# 假设前12列是特征，第13列是目标变量
X = data.iloc[:, 1:11].values
y = data.iloc[:, 13].values

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
  
# 4. 创建基学习器  
# AdaBoost.R2通常与决策树回归器一起使用，但也可以是其他回归模型  
base_estimator = DecisionTreeRegressor(max_depth=20)  
  
# 5. 创建AdaBoost回归模型  
# 你可以通过修改n_estimators参数来改变弱学习器的数量  
ada_reg = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=42)  
  
# 6. 训练模型  
ada_reg.fit(X_train, y_train)  
  
# 7. 进行预测  
y_pred = ada_reg.predict(X_test)  
  
# 8. 评估模型  
mse = mean_squared_error(y_test, y_pred)  
mae = mean_absolute_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print(f"Mean absolute Error: {mae}")  
print(f"Mean Squared Error: {mse}")  
print(f"R^2 Score: {r2}")