# import streamlit as st
# import numpy as np
# import pickle

# # Load model
# model = pickle.load(open("model/house_model.pkl", "rb"))

# st.title("🏠 House Price Prediction")
# st.write("Enter details below to estimate the price.")

# col1, col2 = st.columns(2)

# with col1:
#     area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1200)
#     age = st.number_input("House Age (years)", min_value=0, max_value=100, value=5)

# with col2:
#     bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
#     bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# if st.button("Predict Price"):
#     data = np.array([[area, bedrooms, bathrooms, age]])
#     prediction = model.predict(data)[0]

#     st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)}")



# import streamlit as st
# import numpy as np
# import pickle
# import pandas as pd
# import json
# import os


# st.set_page_config(page_title="House Price App", layout="wide")

# # -------- LOAD MODEL + DATA + METRICS ----------
# model = pickle.load(open("model/house_model.pkl", "rb"))
# metrics = json.load(open("model/metrics.json"))
# data = pd.read_csv("data/house.csv")

# feature_columns = metrics["feature_columns"]

# st.title("🏠 House Price Prediction App")

# # ============= SIDEBAR NAV =============
# page = st.sidebar.radio(
#     "Navigation",
#     ["🔮 Predict Price", "📊 Charts", "✅ Model Accuracy"]
# )

# # ======================================
# # 🔮 PREDICTION PAGE
# # ======================================
# if page == "🔮 Predict Price":
#     st.subheader("Enter house details")

#     col1, col2 = st.columns(2)

#     with col1:
#         area = st.number_input("Area (sq ft)", 300, 10000, 1200)
#         bedrooms = st.number_input("Bedrooms", 1, 10, 3)

#     with col2:
#         bathrooms = st.number_input("Bathrooms", 1, 10, 2)
#         age = st.number_input("House Age (years)", 0, 100, 5)

#     locations = sorted(data["location"].unique())
#     location = st.selectbox("Location", locations)

#     # --------- BUILD FEATURE VECTOR ----------
#     # base numeric inputs
#     inputs = {
#         "area": area,
#         "bedrooms": bedrooms,
#         "bathrooms": bathrooms,
#         "age": age
#     }

#     # start zero vector
#     x = np.zeros(len(feature_columns))

#     # fill numeric
#     for i, col in enumerate(feature_columns):
#         if col in inputs:
#             x[i] = inputs[col]

#         # one-hot for location
#         if col == f"location_{location}":
#             x[i] = 1

#     if st.button("Predict Price"):
#         price = model.predict([x])[0]
#         st.success(f"💰 Estimated Price: ₹ {round(price, 2)}")


# # ======================================
# # 📊 CHARTS PAGE
# # ======================================
# elif page == "📊 Charts":
#     st.subheader("📈 Data Visualizations")

#     st.write("Area vs Price")
#     st.line_chart(data[["area", "price"]])

#     st.write("Average Price by Bedrooms")
#     st.bar_chart(data.groupby("bedrooms")["price"].mean())


# # ======================================
# # ✅ ACCURACY PAGE
# # ======================================
# elif page == "✅ Model Accuracy":
#     st.subheader("📌 Model Performance")

#     st.metric("R2 Score", round(metrics["r2"], 3))
#     st.metric("RMSE", round(metrics["rmse"], 2))

#     st.info("✔ Higher R2 is better\n✔ Lower RMSE is better")



import streamlit as st
import numpy as np
import pickle
import pandas as pd
import json
import os

st.set_page_config(page_title="House Price App", layout="wide")

# ---- THEME STYLE ----
st.markdown(
    """
    <style>
        .main { background-color: #0f172a; color: white; }
        .stButton>button {
            background: #4f46e5;
            color: white;
            border-radius: 8px;
            height: 3rem;
            width: 12rem;
            font-weight: bold;
        }
        .stMetric {
            background: #111827;
            padding: 12px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- LOAD MODEL + DATA + METRICS ----------
model = pickle.load(open("model/house_model.pkl", "rb"))
metrics = json.load(open("model/metrics.json"))
data = pd.read_csv("data/house.csv")

feature_columns = metrics["feature_columns"]

st.title("🏠 House Price Prediction App")

# ============= SIDEBAR NAV =============
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Predict Price", "📊 Charts", "✅ Model Accuracy"]
)

# ======================================
# 🔮 PREDICTION PAGE
# ======================================
if page == "🔮 Predict Price":
    st.subheader("Enter house details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", 300, 10000, 1200)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)

    with col2:
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        age = st.number_input("House Age (years)", 0, 100, 5)

    locations = sorted(data["location"].unique())
    location = st.selectbox("Location", locations)

    # --------- BUILD FEATURE VECTOR ----------
    inputs = {"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age": age}
    x = np.zeros(len(feature_columns))

    for i, col in enumerate(feature_columns):
        if col in inputs:
            x[i] = inputs[col]
        if col == f"location_{location}":
            x[i] = 1

    if st.button("Predict Price"):
        price = model.predict([x])[0]
        st.success(f"💰 Estimated Price: ₹ {round(price, 2)}")

        # ----- CREATE REPORT -----
        report = pd.DataFrame([{
            "Area (sqft)": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age (years)": age,
            "Location": location,
            "Predicted Price (₹)": round(price, 2)
        }])

        st.download_button(
            label="⬇️ Download Report",
            data=report.to_csv(index=False),
            file_name="house_price_report.csv",
            mime="text/csv"
        )

        # -------- SAVE HISTORY --------
        history_row = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age": age,
            "location": location,
            "predicted_price": round(price, 2)
        }

        if os.path.exists("model/history.csv"):
            old = pd.read_csv("model/history.csv")
            new = pd.concat([old, pd.DataFrame([history_row])], ignore_index=True)
            new.to_csv("model/history.csv", index=False)
        else:
            pd.DataFrame([history_row]).to_csv("model/history.csv", index=False)

# ======================================
# 📊 CHARTS PAGE
# ======================================
elif page == "📊 Charts":
    st.subheader("📈 Data Visualizations")

    st.write("Area vs Price")
    st.line_chart(data[["area", "price"]])

    st.write("Average Price by Bedrooms")
    st.bar_chart(data.groupby("bedrooms")["price"].mean())

# ======================================
# ✅ ACCURACY PAGE
# ======================================
elif page == "✅ Model Accuracy":
    st.subheader("📌 Model Performance")

    st.metric("R2 Score", round(metrics["r2"], 3))
    st.metric("RMSE", round(metrics["rmse"], 2))

    st.subheader("📜 Prediction History")

    if os.path.exists("model/history.csv"):
        hist = pd.read_csv("model/history.csv")
        st.dataframe(hist)
    else:
        st.info("No predictions saved yet.")
