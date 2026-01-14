import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# Features and target
X = df.drop(["Price", "Id"], axis=1)
y = df["Price"]

# Log-transform target to stabilize variance
y_log = np.log1p(y)   # log(1+Price)

# Categorical and numeric columns
categorical_cols = ["Location", "Condition", "Garage"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict (remember to inverse log-transform)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # inverse of log1p

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred_log))  # evaluate on log scale
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_log))*100)
