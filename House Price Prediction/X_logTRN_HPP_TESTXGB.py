import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# 1. Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# 2. Feature engineering
df["Age"] = 2025 - df["YearBuilt"]
df["PricePerSqFt"] = df["Price"] / df["Area"]
df["LocationAvgPrice"] = df.groupby("Location")["Price"].transform("mean")
print(df)
# 3. Define features and target
X = df.drop(["Price", "Id"], axis=1)
y = df["Price"]

# 4. Log-transform target for training stability
y_log = np.log1p(y)

# 5. Identify categorical and numeric columns
categorical_cols = ["Location", "Condition", "Garage"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 6. Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# 7. Build model pipeline
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

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 9. Fit model
model.fit(X_train, y_train)

# 10. Predict
y_pred_log = model.predict(X_test)

# Inverse transform predictions and test target back to original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# 11. Evaluate on original scale
print("R2 Score:", r2_score(y_test_original, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_original, y_pred)))
