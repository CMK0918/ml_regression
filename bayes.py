import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization, UtilityFunction

# load data
data = pd.read_csv("boston_house_prices.csv")
X=(data.iloc[:, :13]).astype("float32")
y=(data.iloc[:, 13]).astype("float32")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model build
rf = RandomForestRegressor(max_depth=12, n_estimators=189, random_state=1)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = rf.score(X_train, y_train)
test_r2 = rf.score(X_test,y_test)
cross_r2 = cross_val_score(rf, X_train, y_train, cv=5).mean()

print("train_r2: ", train_r2)
print("cross_r2: ", cross_r2)
print("test_r2: ", test_r2)
print("train mse: ", train_mse)
print("test_mse: ", test_mse)



# def RF_evaluate(n_estimators, max_depth):
#     # Ensure integer values for parameters that require it
#     n_estimators = int(n_estimators)
#     max_depth = int(max_depth)
    
#     # Use cross-validation and return the mean score
#     rf = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             random_state=1,
#             n_jobs=1
#         )
#     rf.fit(X_train, y_train)
#     val = cross_val_score(rf, X_train, y_train, cv=5).mean()
#     return val

# pbounds = {"n_estimators": (20, 250),
#            "max_depth": (5, 12)}

# RF_bo = BayesianOptimization(
#     f=RF_evaluate,
#     pbounds=pbounds,
#     verbose=2,
#     random_state=1
# )

# # 设置高斯过程参数（可选）
# RF_bo.set_gp_params(alpha=1e-3)

# # 创建采集函数
# utility = UtilityFunction(kind="ei", kappa=2.5)

# # 开始优化
# RF_bo.maximize(
#     init_points=5,
#     n_iter=50,
#     acquisition_function=utility
# )

# # 输出最佳结果
# print("最佳参数组合和得分：", RF_bo.max)
# params_max = RF_bo.max['params']










# # define evaluation function
# def RF_evaluate(n_estimators, min_samples_split, max_features, max_depth):
#     # Ensure integer values for parameters that require it
#     n_estimators = int(n_estimators)
#     min_samples_split = int(min_samples_split)
#     max_depth = int(max_depth)
    
#     # Use cross-validation and return the mean score
#     val = cross_val_score(
#         RandomForestRegressor(
#             n_estimators=n_estimators,
#             min_samples_split=min_samples_split,
#             max_features=max_features,
#             max_depth=max_depth,
#             random_state=2,
#             n_jobs=1
#         ),
#         X_train, y_train, cv=5
#     ).mean()
#     return val