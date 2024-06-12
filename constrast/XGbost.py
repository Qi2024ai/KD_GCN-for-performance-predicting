import pandas as pd  
from sklearn.model_selection import train_test_split  
import xgboost as xgb  
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
# 4. 创建XGBoost的DMatrix对象  
dtrain = xgb.DMatrix(X_train, label=y_train)  
dtest = xgb.DMatrix(X_test, label=y_test)  
  
# 5. 设置XGBoost的参数  
param = {  
    'max_depth': 25,  # 最大深度  
    'eta': 0.1,  # 学习率  
    'objective': 'reg:squarederror',  # 目标函数  
    'eval_metric': 'rmse'}  # 评估指标  
  
# 6. 训练模型  
num_round = 800  # 迭代次数  
bst = xgb.train(param, dtrain, num_round)  
  
# 7. 进行预测  
y_pred = bst.predict(dtest)  
  
# 8. 评估模型  
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
  
print(f"Mean Squared Error: {mse}")  
print(f"Mean absolute Error: {mae}")  
print(f"R^2 Score: {r2}")  
  
# 输出模型的特征重要性  
# importances = bst.get_fscore()  
# print("Feature Importances:")  
# for feature, importance in importances.items():  
#     print(f"{feature}: {importance}")