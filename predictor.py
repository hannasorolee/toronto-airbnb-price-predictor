import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# --- Load assets ---
xgb = joblib.load("xgb_model_toronto.joblib")
feature_columns = joblib.load("feature_columns.joblib")
neighborhoods = joblib.load("neighborhoods.joblib")
accommodates_options = joblib.load("accommodates.joblib")
room_type = joblib.load("room_type.joblib")
avg_values = joblib.load("average_values.joblib")

# --- Title ---
st.title("🏙️ Toronto Airbnb Price Estimator")
st.markdown("Estimate the nightly price of your property based on **Toronto Airbnb** data.")

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Customize Your Property")

neigh = st.sidebar.selectbox("Neighbourhood", neighborhoods)
accom = st.sidebar.selectbox("Accommodates", accommodates_options)
#rtype = st.sidebar.selectbox("Room Type", room_type)

default_index = room_type.index("Private room") if "Private room" in room_type else 0
rtype = st.sidebar.selectbox("Room Type", room_type, index=default_index)


bathrooms = st.sidebar.slider("🛁 Bathrooms", 1.0, 4.0, 1.0, 0.5)
host_is_superhost = st.sidebar.checkbox("⭐ Superhost?", value=False)
minimum_nights = st.sidebar.number_input("📆 Minimum Nights", min_value=1, max_value=90, value=14)
number_of_reviews = st.sidebar.slider("📝 Number of Reviews", 0, 500, 50)
review_scores_rating = st.sidebar.slider("🌟 Review Score Rating", 60, 100, 90)

# --- Build Input Vector ---
user_input = {
    "bathrooms": bathrooms,
    "host_is_superhost": int(host_is_superhost),
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "review_scores_rating": review_scores_rating,
    f"neighbourhood_cleansed_{neigh}": 1,
    f"accommodates_{accom}": 1,
    f"room_type_{rtype}": 1
}

input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --- Predict Price ---
price = xgb.predict(input_df)[0]
st.markdown(
    f"""
    <div style="background-color:#d4edda;padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:#155724;">💰 Estimated Nightly Price: <b>${price:.2f}</b></h2>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Feature Impact Simulator ---
st.markdown("### 💡 Feature Impact Simulator")

# 1. Room Type Change (Private → Entire)
if "room_type_Private room" in input_df.columns and input_df["room_type_Private room"].values[0] == 1:
    sim_input = input_df.copy()
    sim_input["room_type_Private room"] = 0  # simulate switching to entire room
    delta_room_type = xgb.predict(sim_input)[0] - price
    st.metric("🏠 Switch to Entire Place", f"${delta_room_type:+.2f}")

# 2. Add +1 Accommodates
next_accom = accom + 1
accom_col = f"accommodates_{next_accom}"
if accom_col in feature_columns:
    sim_input = input_df.copy()
    sim_input[accom_col] = 1
    delta_accom = xgb.predict(sim_input)[0] - price
    st.metric("🧍 Add 1 More Guest", f"${delta_accom:+.2f}")

# 3. Move to Waterfront Island
if "neighbourhood_cleansed_Waterfront Communities-The Island" in input_df.columns and input_df["neighbourhood_cleansed_Waterfront Communities-The Island"].values[0] == 0:
    sim_input = input_df.copy()
    sim_input["neighbourhood_cleansed_Waterfront Communities-The Island"] = 1
    delta_waterfront = xgb.predict(sim_input)[0] - price
    st.metric("🌊 Move to Waterfront Island", f"${delta_waterfront:+.2f}")

# 4. Reduce Minimum Night by 1
sim_input = input_df.copy()
sim_input["minimum_nights"] = max(minimum_nights - 1, 1)
delta_min_night = xgb.predict(sim_input)[0] - price
st.metric("📅 -1 Minimum Night", f"${delta_min_night:+.2f}")

# --- Comparison to Averages ---
st.markdown("### 📊 Your Property vs. Average")

st.markdown(f"**🛁 Bathrooms**: You = {bathrooms}, Avg = {avg_values['bathrooms']:.1f}")
st.markdown(f"**📆 Minimum Nights**: You = {minimum_nights}, Avg = {avg_values['minimum_nights']:.1f}")
st.markdown(f"**📝 Reviews**: You = {number_of_reviews}, Avg = {avg_values['number_of_reviews']:.0f}")
st.markdown(f"**⭐ Superhost**: {'Yes' if host_is_superhost else 'No'}")

# --- Monthly Revenue ---
st.markdown("### 📆 Monthly Revenue Projection")
est_occupancy = st.slider("Estimated Occupancy (%)", 30, 100, 70)
projected_revenue = price * (est_occupancy / 100) * 30
st.success(f"📊 Projected Monthly Revenue: **${projected_revenue:.2f}**")
