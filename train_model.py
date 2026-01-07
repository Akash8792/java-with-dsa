# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import pickle
# import os

# # Load data
# data = pd.read_csv("data/house.csv")

# # Features and target
# X = data[["area", "bedrooms", "bathrooms", "age"]]
# y = data["price"]

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Ensure model folder exists
# os.makedirs("model", exist_ok=True)

# # Save model
# pickle.dump(model, open("model/house_model.pkl", "wb"))

# print("✅ Model trained and saved at model/house_model.pkl")


# 



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import numpy as np
import json

# ---- LOAD DATA ----
data = pd.read_csv("data/house.csv")

# One-hot encode locations + select features
X = pd.get_dummies(
    data[["area", "bedrooms", "bathrooms", "age, location".replace(",", "")]],
    drop_first=True
)

y = data["price"]

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- TRAIN ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- EVALUATE ----
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)

# ---- SAVE MODEL + METRICS ----
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/house_model.pkl", "wb"))

metrics = {
    "r2": float(r2),
    "rmse": float(rmse),
    "feature_columns": list(X.columns)
}

json.dump(metrics, open("model/metrics.json", "w"))

print("Model + metrics saved ✔")
