# =========================================
# Regression Model Comparison Dashboard
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# Safe imports
try:
    from sklearn.svm import SVR
    svr_available = True
except:
    svr_available = False

try:
    from sklearn.neural_network import MLPRegressor
    nn_available = True
except:
    nn_available = False

try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Regression Model Comparison", layout="wide")

st.title("📊 Regression Model Comparison Dashboard")
st.write("Compare multiple regression models on Mall Customers dataset")

# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("Mall_Customers.csv")

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# =========================================
# Preprocessing
# =========================================
le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])

X = df[["Age", "Annual Income (k$)", "Genre"]]
y = df["Spending Score (1-100)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# Models
# =========================================
models = {
    "Linear Regression": LinearRegression(),
    "Stochastic (SGD)": SGDRegressor(max_iter=1000, tol=1e-3),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

if svr_available:
    models["SVR"] = SVR(kernel="rbf")

if nn_available:
    models["Neural Network"] = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=3000)

if xgb_available:
    models["XGBoost"] = XGBRegressor(n_estimators=100, verbosity=0)

# Polynomial Regression
models["Polynomial Regression"] = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

# =========================================
# Training
# =========================================
results = []

for name, model in models.items():

    if name in ["KNN", "SVR", "Neural Network", "Stochastic (SGD)"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append((name, r2, rmse))

# =========================================
# Results
# =========================================
results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE"])
results_df = results_df.sort_values(by="R2 Score", ascending=False)

st.subheader("📊 Model Performance Comparison")
st.dataframe(results_df)

# =========================================
# Visualization (FIXED)
# =========================================
st.subheader("📈 Model Comparison Chart")

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(results_df["Model"], results_df["R2 Score"])
ax.set_xlabel("R2 Score")
ax.set_title("Model Comparison (Higher is Better)")
ax.invert_yaxis()

st.pyplot(fig)   # ✅ FIXED (no plt.show())

# =========================================
# Footer
# =========================================
st.write("🚀 Built for Machine Learning Model Comparison")














