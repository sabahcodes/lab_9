
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

# (a) 80/20 split, then baseline CV RMSE on the training rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))

folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"
baseline_rmse = -cross_val_score(LinearRegression(), X_train, y_train, cv=folds, scoring=scoring)
print(f"Linear baseline mean CV RMSE: {baseline_rmse.mean():.2f} MPa")

# (b) Polynomial + scaling + Ridge grid search
pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])
grid = {"poly__degree": [1, 2, 3], "ridge__alpha": [0.01, 0.1, 1, 10, 100]}
search = GridSearchCV(pipe, grid, cv=folds, scoring=scoring, refit=True)
search.fit(X_train, y_train)

results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
print("\nTop 3 settings:")
for _, row in results.head(3).iterrows():
    print(f"  Rank {int(row['rank_test_score'])}: {row['params']} -> mean CV RMSE {-row['mean_test_score']:.2f} MPa")
print("Best parameters:", search.best_params_)

# (c) Test-set evaluation of the refitted winner
pred = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, pred))
test_mae = mean_absolute_error(y_test, pred)
print(f"\nTest RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE:  {test_mae:.2f} MPa")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.6, label="Test samples")
lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
plt.plot([lo, hi], [lo, hi], "r--", label="y = x")
plt.xlabel("Actual strength (MPa)")
plt.ylabel("Predicted strength (MPa)")
plt.title("Concrete Strength: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=150)
plt.show()

# (d) Baseline vs tuned
print(f"\nBaseline CV RMSE: {baseline_rmse.mean():.2f} | Tuned CV RMSE: {-search.best_score_:.2f}")
