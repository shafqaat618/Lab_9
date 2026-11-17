import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load data and remove identical duplicate rows
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# Keep a separate test set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("Rows after duplicate removal:", len(concrete))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# Same five shuffled folds are used for both models
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

# Linear baseline
baseline = LinearRegression()
baseline_rmse = -cross_val_score(
    baseline, X_train, y_train, cv=folds, scoring=scoring
)
print("Linear baseline mean CV RMSE:", round(baseline_rmse.mean(), 2), "MPa")

# Polynomial features, scaling and Ridge regression
ridge_pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge())
])

parameters = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}
search = GridSearchCV(
    ridge_pipeline, parameters, cv=folds,
    scoring=scoring, refit=True
)
search.fit(X_train, y_train)
print("Best settings:", search.best_params_)
print("Best mean CV RMSE:", round(-search.best_score_, 2), "MPa")
# Show the three highest-ranked parameter combinations
results = pd.DataFrame(search.cv_results_)
top_three = results.sort_values("rank_test_score").head(3)
print("\\nTop three settings:")
for _, row in top_three.iterrows():
    print(
        "Degree:", row["param_poly__degree"],
        "| Alpha:", row["param_ridge__alpha"],
        "| Mean CV RMSE:", round(-row["mean_test_score"], 2), "MPa"
    )
# Evaluate the refitted best model only once on held-out test data
predictions = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
test_mae = mean_absolute_error(y_test, predictions)
print("Test RMSE:", round(test_rmse, 2), "MPa")
print("Test MAE:", round(test_mae, 2), "MPa")
# Actual vs predicted plot with y=x reference line
plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.6)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], "r--", label="Perfect prediction (y = x)")
plt.xlabel("Actual concrete strength (MPa)")
plt.ylabel("Predicted concrete strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=300)
plt.show()
