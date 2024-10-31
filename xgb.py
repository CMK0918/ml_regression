import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# 數據準備
# 載入資料
data=pd.read_csv("boston_house_prices.csv")
#print(df)
X=(data.iloc[:, :13]).astype("float32")
y=(data.iloc[:, 13]).astype("float32")

print(data.info())

fig, ax1 = plt.subplots(7,2, figsize=(20,25))
k = 0
columns = list(data.columns)
for i in range(7):
    for j in range(2):
            sns.distplot(data[columns[k]], ax = ax1[i][j], color = 'green')
            ax1[i][j].grid(True)
            k += 1
plt.show()

def log_transform(col):
    return np.log(col[0])

data["DIS"]=data[["DIS"]].apply(log_transform, axis=1)
#Plot
sns.distplot(data["DIS"], color = 'green')
plt.grid(True)
plt.show()

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

# 標準化數據
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def xgb_model() -> dict:
    '''
    this function build XGBRegressor model
    claculate train R2, cross_validation R2, test R2, test MAE, y_pred 
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        xgb = XGBRegressor()
        xgb.fit(X_train,y_train)

        # model evaluate
        y_pred = xgb.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        train_r2 = xgb.score(X_train, y_train)
        test_r2 = xgb.score(X_test, y_test)
        cross_r2 = cross_val_score(xgb, X_train, y_train, cv=5).mean()

        # append result to the list
        result["y_pred"] = y_pred
        result["mse"] = mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        result["cross_r2"] = cross_r2
        return result
    
    except Exception as e:
        print(f"xgb_model error {e}")

xgb_model_dict = xgb_model()
print("xgb train R2:", xgb_model_dict["train_r2"])
print("xgb cross R2: ",xgb_model_dict["cross_r2"])
print("xgb test R2: ", xgb_model_dict["test_r2"])
print("xgb MSE: ", xgb_model_dict["mse"])
