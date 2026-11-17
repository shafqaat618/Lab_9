import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
homes = pd.read_csv("real_estate.csv")

# Convert distance from metres to units of 100 metres
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100

X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)
print("Slope:", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))
# Predict price per unit area at 500 metres (5 units of 100 metres)
new_home = pd.DataFrame({"distance_100m": [5]})
prediction = model.predict(new_home)[0]
print("Predicted price at 500 m:", round(prediction, 2), "price units")

# Plot observations and fitted regression line
plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, alpha=0.5, label="Actual observations")
line_x = pd.DataFrame({
    "distance_100m": sorted(X["distance_100m"].unique())
})
plt.plot(line_x["distance_100m"], model.predict(line_x),
         color="red", linewidth=2, label="Fitted regression line")
plt.xlabel("Distance to nearest MRT station (100 metres)")
plt.ylabel("House price of unit area (10,000 NTD per ping)")
plt.title("Housing Price vs Distance to Metro Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=300)
plt.show()
