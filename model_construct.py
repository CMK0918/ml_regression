import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
import matplotlib.pyplot as plt

# 數據準備
# 載入資料
df=pd.read_csv("boston_house_prices.csv")
#print(df)
X=(df.iloc[:, :13]).astype("float32")
y=(df.iloc[:, 13]).astype("float32")

# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# RandomForest model build

def rf_model(treenumber=150, depth=20) -> dict:
    '''
    this function build RandmoForestRegressor model
    claculate train R2, cross_validation R2, test R2, test MAE, y_pred 
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        rf = RandomForestRegressor(n_estimators=treenumber, max_depth=depth, random_state=0)
        rf.fit(X_train,y_train)

        # model evaluate
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)
        cross_r2 = cross_val_score(rf, X_train, y_train, cv=5).mean()

        # append result to the list
        result["y_pred"] = y_pred
        result["mse"] = mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        result["cross_r2"] = cross_r2
        return result
    
    except Exception as e:
        print(f"rf_model error {e}")

def gbr_model(treenumber=100, learnrate=0.2) -> dict:
    '''
    this function build GradientBoostingRegressor model
    and return train R2, cross_validation R2, test R2, test MAE, y_pred     
    '''
    try:

        # initial result list
        result = {}

        # model fit
        gbr = GradientBoostingRegressor(n_estimators=treenumber ,learning_rate=learnrate, random_state=0)
        gbr.fit(X_train,y_train)

        # model evaluate
        y_pred = gbr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        train_r2 = gbr.score(X_train, y_train)
        test_r2 = gbr.score(X_test, y_test)
        cross_r2 = cross_val_score(gbr, X_train, y_train, cv=5).mean()

        # append result to the list
        result["y_pred"] = y_pred
        result["mse"] = mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        result["cross_r2"] = cross_r2
        return result

    except Exception as e:
        print(f"gbr error {e}")
    




rf_model_dict = rf_model(treenumber=60, depth=30)
print("RF train R2:", rf_model_dict["train_r2"])
print("RF cross R2: ",rf_model_dict["cross_r2"])
print("RF test R2: ", rf_model_dict["test_r2"])
print("RF MSE: ", rf_model_dict["mse"])


gbr_model_dict = gbr_model()
print("GBR train R2:", gbr_model_dict["train_r2"])
print("GBR cross R2: ",gbr_model_dict["cross_r2"])
print("GBR test R2: ", gbr_model_dict["test_r2"])
print("GBR MSE: ", gbr_model_dict["mse"])




def dynamicfusion_model() -> dict:
    '''
    dynamic change y_pred[n] and output y_pred and mse
    '''
    try:
        # 計算錯誤
        error_rf = rf_model_dict["y_pred"] - y_test
        error_gbr = gbr_model_dict["y_pred"] - y_test
        result = {}
        # gate function
        y_pred_fussion = []
        for n in range(len(y_test)):
            if abs(error_rf.iloc[n]) - abs(error_gbr.iloc[n]) > 0:
                y_pred_fussion.append(0.2 * rf_model_dict["y_pred"][n] + 0.8 * gbr_model_dict["y_pred"][n])
            elif abs(error_rf.iloc[n]) - abs(error_gbr.iloc[n]) <= 0:
                y_pred_fussion.append(0.8 * rf_model_dict["y_pred"][n] + 0.2 * gbr_model_dict["y_pred"][n])
        
        y_pred_fussion = np.array(y_pred_fussion)
        mse_fussion = mean_squared_error(y_test, y_pred_fussion)

        result["y_pred"] = y_pred_fussion
        result["mse"] = mse_fussion

        return result
    
    except Exception as e:
        print(f"dynamicfusion_model error {e}")

# evaluate model
dynamicfusion_model_dict = dynamicfusion_model()
r2_fussion = r2_score(y_test, dynamicfusion_model_dict["y_pred"])
print("Dynamic Fusion Model MSE:", dynamicfusion_model_dict["mse"])
print("Dynamic Fusion Model R²:", r2_fussion)

# plt.figure(figsize=(20, 5))
# plt.plot(y_test.values, label='True Values', color='blue')
# plt.plot(dynamicfusion_model_dict["y_pred"], label='Fused Predictions', color='orange')
# # plt.plot(rf_model_dict["y_pred"], label="rf Predictions", color="red")
# plt.plot(gbr_model_dict["y_pred"], label="gbr Predictions", color="green")
# plt.title('Comparison of True Values and Fused Predictions')
# plt.xlabel('Sample Index')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

def staticfusion_model() -> dict:
    '''
    
    '''
    try:

        result = {}
        
        y_pred = 0.4 * rf_model_dict["y_pred"] + 0.6 * gbr_model_dict["y_pred"]
        mse = mean_squared_error(y_test, y_pred)

        result["y_pred"] = y_pred
        result["mse"] = mse

        return result
    
    except Exception as e:
        print(f"staticfusion_model error: {e}")

staticfusion_model_dict = staticfusion_model()

r2_fussion = r2_score(y_test, staticfusion_model_dict["y_pred"])
print("Static Fusion Model MSE:", staticfusion_model_dict["mse"])
print("Static Fusion Model R²:", r2_fussion)