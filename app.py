
import os
import streamlit as st
import pandas as pd
import numpy as np
from recommender import CarRecommender
from utils import load_data, apply_filters

# === Configuration ===
DATA_PATH = os.environ.get("CAR_DATA_PATH", "Cleaned_Car_data.csv")

st.set_page_config(page_title="Car Recommendation System", page_icon="🚗", layout="wide")
st.title("🚗 Car Recommendation System")
st.caption("Content-based recommendations using TF‑IDF + k‑NN (cosine)")

# Load data
@st.cache_data(show_spinner=False)
def get_data(path):
    return load_data(path)

df = get_data(DATA_PATH)

# Sidebar filters
st.sidebar.header("Filters")
min_price = int(df["Price"].min())
max_price = int(df["Price"].max())
price_range = st.sidebar.slider("Price range (₹)", min_price, max_price, (min_price, max_price), step=10000)
companies = ["All"] + sorted(df["company"].unique().tolist())
fuels = ["All"] + sorted(df["fuel_type"].unique().tolist())
company = st.sidebar.selectbox("Company", companies)
fuel = st.sidebar.selectbox("Fuel Type", fuels)

min_year = int(df["year"].min())
max_year = int(df["year"].max())
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year), step=1)

max_kms_input = st.sidebar.number_input("Max kms driven", min_value=0, max_value=int(df["kms_driven"].max()), value=int(df["kms_driven"].quantile(0.9)), step=1000)

# Build the recommender
@st.cache_resource(show_spinner=False)
def build_recommender(dataframe: pd.DataFrame):
    rec = CarRecommender(max_features=5000).fit(dataframe)
    return rec

rec = build_recommender(df)

# Copy df and apply filters
filtered = apply_filters(
    df,
    min_price=price_range[0], max_price=price_range[1],
    company=company, fuel_type=fuel,
    min_year=year_range[0], max_year=year_range[1],
    max_kms=max_kms_input
)

st.subheader("Find Similar Cars")
col1, col2 = st.columns(2)
with col1:
    example_idx = st.selectbox(
        "Pick a car as the query (from filtered data):",
        options=list(filtered.index),
        format_func=lambda i: f"{filtered.loc[i, 'name'].title()} | {filtered.loc[i, 'company'].title()} | ₹{filtered.loc[i, 'Price']:,} | {int(filtered.loc[i, 'year'])} | {filtered.loc[i, 'fuel_type'].title()}"
    )
with col2:
    top_k = st.number_input("How many recommendations?", min_value=3, max_value=30, value=10, step=1)

if st.button("Recommend similar"):
    if example_idx is not None:
        pairs = rec.recommend_similar_to_index(int(example_idx), k=int(top_k))
        results = rec.df.iloc[[i for i,_ in pairs]].copy()
        results["score"] = [1 - d for _, d in pairs]  # higher is better
        st.markdown("### Results (similarity-based)")
        st.dataframe(results[["name","company","fuel_type","year","kms_driven","Price","score"]].sort_values("score", ascending=False).reset_index(drop=True))

st.divider()

st.subheader("Search by Text + Constraints")
qcol1, qcol2, qcol3 = st.columns(3)
with qcol1:
    text_query = st.text_input("Free text (brand/model/fuel/keywords)", placeholder="e.g., honda diesel suv")
with qcol2:
    q_year = st.number_input("Preferred year (optional)", min_value=min_year, max_value=max_year, value=min_year, step=1)
with qcol3:
    q_kms = st.number_input("Preferred kms (optional)", min_value=0, max_value=int(df['kms_driven'].max()), value=int(df['kms_driven'].median()), step=1000)

if st.button("Search"):
    pairs = rec.recommend_by_query(text_query.strip().lower(), year=int(q_year), kms=int(q_kms), k=int(top_k))
    idxs = [i for i,_ in pairs]
    dists = [d for _,d in pairs]
    results = rec.df.iloc[idxs].copy()
    results["score"] = [1 - d for d in dists]
    # Apply filters after retrieval for relevance + constraints
    results = results.merge(filtered.reset_index(), on=list(rec.df.columns), how="inner")
    if len(results) == 0:
        st.info("No results matching the filters. Try relaxing filters or a different query.")
    else:
        st.dataframe(results[["name","company","fuel_type","year","kms_driven","Price","score"]].sort_values("score", ascending=False).reset_index(drop=True))

st.caption("Tip: Use the sidebar filters to constrain by budget, brand, fuel type, year, and kms.")
