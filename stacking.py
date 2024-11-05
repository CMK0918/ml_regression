import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error

# load data
data=pd.read_csv("boston_house_prices.csv")
X=(data.iloc[:, :13]).astype("float32")
y=(data.iloc[:, 13]).astype("float32")

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def xgb_model() -> dict:
    '''
    this function build XGBRegressor model
    claculate train R2, test R2, train MAE, test MAE 
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        xgb = XGBRegressor(random_state=1)
        xgb.fit(X_train,y_train)

        # model evaluate
        y_pred_train = xgb.predict(X_train)
        y_pred_test = xgb.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = xgb.score(X_train, y_train)
        test_r2 = xgb.score(X_test, y_test)
        
        # append result to the list
        result["train_mse"] = train_mse
        result["test_mse"] = test_mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    
    except Exception as e:
        print(f"xgb_model error {e}")


def knn_model() -> dict:
    '''
    this function build KNeighborsRegressor model
    claculate train R2, test R2, train MAE, test MAE
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        knn = KNeighborsRegressor()
        knn.fit(X_train,y_train)

        # model evaluate
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = knn.score(X_train, y_train)
        test_r2 = knn.score(X_test, y_test)

        # append result to the list
        result["train_mse"] = train_mse 
        result["test_mse"] = test_mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    
    except Exception as e:
        print(f"knn_model error {e}")


def rf_model() -> dict:
    '''
    this function build RandomForestRegressor model
    claculate train R2, test R2, train MAE, test MAE
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        rf = RandomForestRegressor(random_state=1)
        rf.fit(X_train,y_train)

        # model evaluate
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_test, y_test)

        # append result to the list
        result["train_mse"] = train_mse 
        result["test_mse"] = test_mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    
    except Exception as e:
        print(f"rf_model error {e}")


def stacking_model() -> dict:
    '''
    this function build stacking model
    claculate train R2, test R2, train MAE, test MAE
    '''
    try:
        
        # initial result dict
        result = {}

        esitmators = [
            ("linear", RandomForestRegressor(random_state=1)),
            ("xgb", XGBRegressor(random_state=1)),
            ("knn", KNeighborsRegressor())
        ]

        stack = StackingRegressor(estimators=esitmators, final_estimator=LinearRegression())
        stack.fit(X_train,y_train)

        y_pred_train = stack.predict(X_train)
        y_pred_test = stack.predict(X_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = stack.score(X_train, y_train)
        test_r2 = stack.score(X_test, y_test)

        # append result to the list
        result["train_mse"] = train_mse 
        result["test_mse"] = test_mse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    
    except Exception as e:
        print(f"stacking_model error {e}")

# call model function
xgb_model_dict = xgb_model()
knn_model_dict = knn_model()
rf_model_dict = rf_model()
stacking_model_dict = stacking_model()

# evaluate table
table = pd.DataFrame({
    "train R2":[xgb_model_dict["train_r2"],knn_model_dict["train_r2"],rf_model_dict["train_r2"],stacking_model_dict["train_r2"]],
    "test R2":[xgb_model_dict["test_r2"],knn_model_dict["test_r2"],rf_model_dict["test_r2"],stacking_model_dict["test_r2"]],
    "train mse":[xgb_model_dict["train_mse"],knn_model_dict["train_mse"],rf_model_dict["train_mse"],stacking_model_dict["train_mse"]],
    "test mse":[xgb_model_dict["test_mse"],knn_model_dict["test_mse"],rf_model_dict["test_mse"],stacking_model_dict["test_mse"]]
}, index=["xgboost", "knn", "randomforest", "stacking"])

print(table)





# print("xgb train R2:", xgb_model_dict["train_r2"])
# print("xgb test R2: ", xgb_model_dict["test_r2"])
# print("xgb train MSE: ", xgb_model_dict["train_mse"])
# print("xgb test MSE: ", xgb_model_dict["test_mse"])

# print("knn train R2:", knn_model_dict["train_r2"])
# print("knn test R2: ", knn_model_dict["test_r2"])
# print("knn train MSE: ", knn_model_dict["train_mse"])
# print("knn test MSE: ", knn_model_dict["test_mse"])


# print("rf train R2:", rf_model_dict["train_r2"])
# print("rf test R2: ", rf_model_dict["test_r2"])
# print("rf train MSE: ", rf_model_dict["train_mse"])
# print("rf test MSE: ", rf_model_dict["test_mse"])

# print("stack train R2: ", stacking_model_dict["train_r2"])
# print("stack test R2: ", stacking_model_dict["test_r2"])
# print("stack mse: ", stacking_model_dict["train_mse"])
# print("stack mse: ", stacking_model_dict["test_mse"])