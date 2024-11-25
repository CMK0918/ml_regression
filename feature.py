import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 載入數據集
def load_dataset(data_path: str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(data_path, keep_default_na=False, na_values=["NA"])
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse the dataset at {data_path}")

# 合併訓練測試集做相同數據清洗步驟
def data_clean(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:

    rows_train = df_train.shape[0]

    # Combine Train Set & Test Set
    combined = pd.concat((df_train, df_test), sort=False)
    combined.drop(["SalePrice"], axis=1, inplace=True)
    combined.drop(["Id"], axis=1, inplace=True) 

    # Remove Missing Features > 15% Columns 
    combined.dropna(thresh=len(combined)*0.85, axis=1, inplace=True)
    
    # Fill NA Values
    # Category Front Fill for Missing Values < 10 
    combined["KitchenQual"].fillna(method="ffill", inplace=True)
    combined["Electrical"].fillna(method="ffill", inplace=True)
    combined["SaleType"].fillna(method="ffill", inplace=True)
    combined["Exterior2nd"].fillna(method="ffill", inplace=True)
    combined["Exterior1st"].fillna(method="ffill", inplace=True)
    combined["Utilities"].fillna(method="ffill", inplace=True)
    combined["Functional"].fillna(method="ffill", inplace=True)
    combined["MSZoning"].fillna(method="ffill", inplace=True)

    # Numeric GarageYrBlt Fill Median 1980s
    combined["GarageYrBlt"].fillna(1980, inplace=True)

    # Category fill None
    categorical_cols = combined.select_dtypes(include="object").columns
    combined[categorical_cols] = combined[categorical_cols].fillna("None")
    
    # Numerical fill 0
    numeric_cols = combined.select_dtypes(exclude="object").columns
    combined[numeric_cols] = combined[numeric_cols].fillna(0)
    
    # Category Number to Str
    combined[["MoSold", "MSSubClass", "YrSold"]] = combined[["MoSold", "MSSubClass", "YrSold"]].astype(str)
    
    # # 分割回訓練集和測試集
    train_set = combined.iloc[:rows_train, :]
    test_set = combined.iloc[rows_train:, :]
  
    return train_set, test_set

def split_transform(df_train_features, df_train_target):
    
    # 分割訓練驗證
    X_train, X_valid, y_train, y_valid = train_test_split(df_train_features, df_train_target, test_size=0.1, shuffle=True, random_state=1)

    # 類別型欄位 數值型欄位
    categorical_cols = X_train.select_dtypes(include="object").columns
    numeric_cols = X_train.select_dtypes(exclude="object").columns

    # 定義 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),  # 數值型欄位用 RobustScaler
            ("cat", OneHotEncoder(sparse_output=False,handle_unknown='ignore'), categorical_cols)  # 類別型欄位用 OneHotEncoder
        ]
    )

    # 特徵轉換
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    
    # 目標轉換
    y_train = np.log1p(y_train)
    y_valid = np.log1p(y_valid)

    return X_train_transformed, y_train, X_valid_transformed, y_valid

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
        y_pred_test = rf.predict(X_valid)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_test))
        train_r2 = rf.score(X_train, y_train)
        test_r2 = rf.score(X_valid, y_valid)

        # append result to the list
        result["train_rmse"] = train_rmse 
        result["test_rmse"] = test_rmse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    except Exception as e:
        print(f"rf_model error {e}")

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
        y_pred_test = xgb.predict(X_valid)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_test))
        train_r2 = xgb.score(X_train, y_train)
        test_r2 = xgb.score(X_valid, y_valid)

        # append result to the list
        result["train_rmse"] = train_rmse 
        result["test_rmse"] = test_rmse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    except Exception as e:
        print(f"rf_model error {e}")

def linear_model() -> dict:
    '''
    this function build LinearRegression model
    claculate train R2, test R2, train MAE, test MAE
    '''
    try:
        
        # initial result dict
        result = {}

        # model fit
        lr = LinearRegression()
        lr.fit(X_train,y_train)

        # model evaluate
        y_pred_train = lr.predict(X_train)
        y_pred_test = lr.predict(X_valid)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_test))
        train_r2 = lr.score(X_train, y_train)
        test_r2 = lr.score(X_valid, y_valid)

        # append result to the list
        result["train_rmse"] = train_rmse 
        result["test_rmse"] = test_rmse 
        result["train_r2"] = train_r2
        result["test_r2"] = test_r2
        return result
    except Exception as e:
        print(f"rf_model error {e}")

def evaluate_and_display_models():
    
    xgb_model_dict = xgb_model()
    rf_model_dict = rf_model()
    lr_model_dict = linear_model()
    
    # 設置 pandas 顯示格式為四位小數
    pd.options.display.float_format = '{:.4f}'.format

    # 評估結果表格
    table = pd.DataFrame({
        "train R2": [xgb_model_dict["train_r2"], rf_model_dict["train_r2"], lr_model_dict["train_r2"]],
        "test R2": [xgb_model_dict["test_r2"], rf_model_dict["test_r2"], lr_model_dict["test_r2"]],
        "train RMSE": [xgb_model_dict["train_rmse"], rf_model_dict["train_rmse"], lr_model_dict["train_rmse"]],
        "test RMSE": [xgb_model_dict["test_rmse"], rf_model_dict["test_rmse"], lr_model_dict["test_rmse"]]
    }, index=["xgboost", "randomforest", "linear"])

    # 顯示結果表格
    print(table)


df_train = load_dataset(data_path = "train.csv")
df_test = load_dataset(data_path = "test.csv")
df_train_target = df_train["SalePrice"]
df_train_features, df_test_features = data_clean(df_train, df_test)
X_train, y_train, X_valid, y_valid = split_transform(df_train_features, df_train_target)
evaluate_and_display_models()



def test_submission(df_train_features, df_test_features, df_train_target):
    # 類別型欄位 和 數值型欄位
    categorical_cols = df_train_features.select_dtypes(include="object").columns
    numeric_cols = df_train_features.select_dtypes(exclude="object").columns

    # 定義 ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),  # 數值型欄位用 RobustScaler
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)  # 類別型欄位用 OneHotEncoder
        ]
    )

    # 特徵轉換
    X_train_transformed = preprocessor.fit_transform(df_train_features)
    X_test_transformed = preprocessor.transform(df_test_features)
    
    # 模型擬合
    rf = RandomForestRegressor(random_state=1)
    rf.fit(X_train_transformed, df_train_target)
    
    # 預測
    predictions = rf.predict(X_test_transformed)
    
    # 生成從1461開始的 ID 列
    id_values = range(1461, 1461 + len(df_test_features))  # 從 1461 開始生成 ID

    # 將預測結果存成 submission.csv
    submission = pd.DataFrame({
        'Id': id_values,  # 使用生成的 ID
        'SalePrice': predictions
    })
    
    # 儲存成 CSV 文件
    submission.to_csv('submission.csv', index=False)
    print("Submission saved as 'submission.csv'")

test_submission(df_train_features, df_test_features, df_train_target)