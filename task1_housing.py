#task number 1 . data sci
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# (a) Fit linear regression on all 414 rows
model = LinearRegression()
model.fit(X, y)
print(f"Slope: {model.coef_[0]:.4f} price units per 100 m")
print(f"Intercept: {model.intercept_:.4f} price units")

# (b) Predict the price at 500 m (distance_100m = 5)
new_home = pd.DataFrame({"distance_100m": [5]})
print(f"Predicted price at 500 m: {model.predict(new_home)[0]:.2f} price units")

# (c) Plot observations and fitted line
plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, alpha=0.5, label="Observed transactions")
xs = pd.DataFrame({"distance_100m": sorted(X["distance_100m"])})
plt.plot(xs["distance_100m"], model.predict(xs), color="red", linewidth=2, label="Fitted line")
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("House price per unit area (10,000 NTD/ping)")
plt.title("House Price vs Distance to MRT Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=150)
plt.show()
