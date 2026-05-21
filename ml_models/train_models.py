import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# =========================================================
# LOAD DATASET
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

csv_path = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_mutual_funds.csv"
)

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully!")
print(df.head())


# =========================================================
# CLEAN DATA
# =========================================================

# Replace '-' with NaN
df.replace("-", pd.NA, inplace=True)

# Numeric columns
numeric_cols = [
    "min_sip",
    "min_lumpsum",
    "expense_ratio",
    "fund_size_cr",
    "fund_age_yr",
    "sortino",
    "alpha",
    "sd",
    "beta",
    "sharpe",
    "rating",
    "returns_1yr",
    "returns_3yr",
    "returns_5yr"
]

# Convert to numeric
for col in numeric_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
    )

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Remove missing values
df = df.dropna()

print("\nCleaned Dataset Shape:")
print(df.shape)


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[[
    "min_sip",
    "min_lumpsum",
    "expense_ratio",
    "fund_size_cr",
    "fund_age_yr",
    "sortino",
    "alpha",
    "sd",
    "beta",
    "sharpe",
    "rating"
]]

y = df["returns_5yr"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# =========================================================
# MODELS
# =========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsRegressor()

}


# =========================================================
# TRAINING
# =========================================================

best_model = None
best_score = -999

print("\n================ MODEL RESULTS ================\n")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Print Results
    print(f"{name}")

    print(f"MAE Score : {mae:.2f}")

    print(f"R2 Score  : {r2:.2f}")

    print("-" * 45)

    # Save Best Model
    if r2 > best_score:

        best_score = r2

        best_model = model


# =========================================================
# SAVE BEST MODEL
# =========================================================

model_path = os.path.join(
    BASE_DIR,
    "ml_models",
    "best_return_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nBest Model Saved Successfully!")

print(f"\nBest R2 Score: {best_score:.2f}")

print("\nModel Path:")
print(model_path)