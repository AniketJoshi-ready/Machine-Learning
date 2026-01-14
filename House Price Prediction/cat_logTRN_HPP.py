import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

# 1. Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

# 2. Handle outliers (remove top 1% extreme prices)
df = df[df["Price"] < df["Price"].quantile(0.99)]

# 3. Feature engineering
df["Age"] = 2025 - df["YearBuilt"]
df["PricePerSqFt"] = df["Price"] / df["Area"]
df["LocationAvgPrice"] = df.groupby("Location")["Price"].transform("mean")

# 4. Define features and target
X = df.drop(["Price", "Id"], axis=1)
y = df["Price"]

# 5. Log-transform target for training stability
y_log = np.log1p(y)

# 6. Identify categorical columns
categorical_cols = ["Location", "Condition", "Garage"]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# 8. Build CatBoost model (handles categorical features directly)
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    cat_features=categorical_cols,
    verbose=0
)

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
print("avg price of overall figures of Price column: ",np.average(df["Price"]))
# 12. Cross-validation for stability
scores = cross_val_score(model, X, y_log, cv=5, scoring="r2")
print("Cross-validated R2:", scores.mean())
