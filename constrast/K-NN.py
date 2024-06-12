import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# 读取csv文件
data = pd.read_csv('medium C steel_ori.csv',encoding='gbk')

# 假设前12列是特征，第13列是目标变量
X = data.iloc[:, 1:11].values
y = data.iloc[:, 11].values

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
  
# 5. 创建KNN回归模型  
# 你可以通过修改n_neighbors参数来改变K的值  
knn_reg = KNeighborsRegressor(n_neighbors=3)  
  
# 6. 训练模型  
knn_reg.fit(X_train, y_train)  
  
# 7. 进行预测  
y_pred = knn_reg.predict(X_test)  
  
# 8. 评估模型  

mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  
print(f"Mean absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")  
print(f"R^2 Score: {r2}")