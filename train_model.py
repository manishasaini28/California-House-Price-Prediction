import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("housing.csv")

print("First 5 Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

# Features
X = df.drop("median_house_value", axis=1)

# Target
y = df["median_house_value"]

# Numerical Columns
num_cols = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

# Categorical Column
cat_cols = ["ocean_proximity"]

# Numerical Pipeline
num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical Pipeline
cat_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# Model Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Graph 1 : Income vs House Price

plt.figure(figsize=(8,5))

plt.scatter(df["median_income"],
            df["median_house_value"],
            alpha=0.4)

plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Income vs House Price")

plt.grid(True)

plt.savefig("static/charts/income_vs_price.png")

plt.close()

# Graph 2 : Feature Importance

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance = model.named_steps["regressor"].feature_importances_

plt.figure(figsize=(10,6))

plt.barh(feature_names, importance)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("static/charts/feature_importance.png")

plt.close()

print("\nModel Performance")
print("MAE :", mean_absolute_error(y_test, pred))
print("R2 Score :", r2_score(y_test, pred))

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("\nmodel.pkl Saved Successfully")