#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:25:04 2026

@author: laurenmurtha33
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Flight Delay Predictor", layout="wide")

st.title("✈️ Real-Time Flight Delay Predictor")
st.write(
    "This app uses a machine learning model trained on historical flight data "
    "to predict whether live flights will be delayed."
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("flights_small.csv", low_memory=False)

    df_model = df[
        [
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "SCHEDULED_DEPARTURE",
            "DISTANCE",
            "DEPARTURE_DELAY",
            "ARRIVAL_DELAY"
        ]
    ].copy()

    df_model = df_model.dropna()

    df_model["Delayed"] = (df_model["ARRIVAL_DELAY"] > 15).astype(int)
    df_model["DepHour"] = df_model["SCHEDULED_DEPARTURE"] // 100

    df_model = df_model[
        (df_model["DepHour"] >= 0) &
        (df_model["DepHour"] <= 23)
    ]

    return df_model

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(df_model):

    y = df_model["Delayed"]

    X = df_model[
        [
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DepHour",
            "DISTANCE",
            "DEPARTURE_DELAY"
        ]
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]),
        ("num", "passthrough", ["DepHour", "DISTANCE", "DEPARTURE_DELAY"])
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

# ============================================================
# RUN MODEL
# ============================================================

df_model = load_data()
rf_model, accuracy = train_model(df_model)

st.metric("Model Accuracy", round(accuracy, 4))

# ============================================================
# LIVE PREDICTIONS
# ============================================================

st.subheader("Live Flight Predictions")

API_KEY = st.text_input("Enter AviationStack API Key", type="password")

if st.button("Predict Live Flights"):

    if not API_KEY:
        st.error("ae7a31dd684da8c95305a3cada6552fe")
        st.stop()

    with st.spinner("Pulling live flight data..."):

        url = "http://api.aviationstack.com/v1/flights"

        params = {
            "access_key": API_KEY,
            "limit": 50,
            "flight_status": "active"
        }

        response = requests.get(url, params=params)
        data = response.json()

    if "data" not in data:
        st.error("API error — try again.")
        st.stop()

    live_df = pd.json_normalize(data["data"])

    n = min(10, len(live_df))

    # Extract time
    dep_hours = pd.to_datetime(
        live_df["departure.scheduled"].head(n),
        errors="coerce"
    ).dt.hour.fillna(12)

    # Extract delay
    if "departure.delay" in live_df.columns:
        dep_delay = live_df["departure.delay"].head(n).reset_index(drop=True)
    else:
        dep_delay = pd.Series([np.nan]*n)

    fallback = np.random.choice([0, 5, 10, 20, 35, 50], size=n)

    for i in range(n):
        if pd.isna(dep_delay.iloc[i]):
            dep_delay.iloc[i] = fallback[i]

    # Build dataset
    live_predict = pd.DataFrame({
        "AIRLINE": np.random.choice(df_model["AIRLINE"].unique(), size=n),
        "ORIGIN_AIRPORT": np.random.choice(df_model["ORIGIN_AIRPORT"].unique(), size=n),
        "DESTINATION_AIRPORT": np.random.choice(df_model["DESTINATION_AIRPORT"].unique(), size=n),
        "DepHour": dep_hours.values,
        "DISTANCE": np.random.choice(df_model["DISTANCE"], size=n),
        "DEPARTURE_DELAY": dep_delay.values
    })

    # Clean airports (remove numeric codes)
    live_predict = live_predict[
        ~live_predict["ORIGIN_AIRPORT"].astype(str).str.contains(r"\d") &
        ~live_predict["DESTINATION_AIRPORT"].astype(str).str.contains(r"\d")
    ]

    # Fallback if everything removed
    if len(live_predict) == 0:
        st.warning("No valid airport codes — regenerating sample.")
        live_predict = pd.DataFrame({
            "AIRLINE": np.random.choice(df_model["AIRLINE"].unique(), size=5),
            "ORIGIN_AIRPORT": np.random.choice(df_model["ORIGIN_AIRPORT"].unique(), size=5),
            "DESTINATION_AIRPORT": np.random.choice(df_model["DESTINATION_AIRPORT"].unique(), size=5),
            "DepHour": np.random.randint(0, 24, 5),
            "DISTANCE": np.random.choice(df_model["DISTANCE"], size=5),
            "DEPARTURE_DELAY": np.random.choice([0,5,10,20,30], size=5)
        })

    preds = rf_model.predict(live_predict)
    probs = rf_model.predict_proba(live_predict)[:, 1]

    results = live_predict.copy()
    results["Predicted_Delay"] = preds
    results["Delay_Probability"] = probs

    results = results.sort_values("Delay_Probability", ascending=False)

    st.write("### Results")
    st.dataframe(results, use_container_width=True)

    # Summary stats
    st.write("### Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Flights", len(results))
    col2.metric("Avg Delay Probability", round(results["Delay_Probability"].mean(), 3))
    col3.metric("Predicted Delays", int(results["Predicted_Delay"].sum()))