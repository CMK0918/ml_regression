import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

# 載入資料
df = pd.read_csv("boston_house_prices.csv")
X = df.iloc[:, :13].astype("float32")
y = df.iloc[:, 13].astype("float32")

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 將 y_train 和 y_test 轉換為 Numpy 陣列
y_train = y_train.values
y_test = y_test.values

# 標準化數據
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LMS算法參數
learning_rate = 0.001  # 調整學習率
epochs = 500           # 減少訓練次數以避免過度擬合
n_features = X_train.shape[1]
weights = np.random.randn(n_features) * 0.01  # 初始化權重為較小隨機值
bias = 0  # 初始化偏置

# 訓練LMS算法
for epoch in range(epochs):
    for i in range(len(X_train)):
        prediction = np.dot(X_train[i], weights) + bias  # 預測值
        error = y_train[i] - prediction                  # 計算誤差
        weights += learning_rate * error * X_train[i]    # 更新權重
        bias += learning_rate * error                    # 更新偏置

# 在測試集上進行預測
y_pred = np.dot(X_test, weights) + bias

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print("測試集上的均方誤差 (MSE):", mse)
print("測試集 R²:", r2_score(y_test,y_pred))

#=================================================================================
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("線性測試集MSE: ", mse)
print("線性測試集R2: ", r2_score(y_test, y_pred))