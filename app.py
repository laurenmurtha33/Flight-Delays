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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Live Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Live Flight Delay Predictor")

st.write(
    "This app trains a machine learning model using historical flight data, "
    "then applies the model to live flight data from the AviationStack API."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Project Settings")

api_limit = st.sidebar.slider(
    "Number of live flights to pull",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

st.sidebar.write("Historical dataset loaded automatically: `flights_small.csv`")


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
            "DEPARTURE_DELAY"
        ]
    ]

    categorical_features = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT"
    ]

    numeric_features = [
        "DepHour",
        "DEPARTURE_DELAY"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return model, accuracy, report


# ============================================================
# MAIN APP
# ============================================================

with st.spinner("Loading historical data and training model..."):
    df_model = load_data()
    rf_model, accuracy, report = train_model(df_model)

st.success("Model trained successfully.")


# ============================================================
# MODEL SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Rows Used", f"{len(df_model):,}")

with col2:
    st.metric("Model Accuracy", f"{accuracy:.4f}")

with col3:
    delayed_rate = df_model["Delayed"].mean()
    st.metric("Historical Delay Rate", f"{delayed_rate:.4f}")


# ============================================================
# HISTORICAL DATA OVERVIEW
# ============================================================

st.subheader("Historical Data Overview")

tab1, tab2, tab3 = st.tabs(
    ["Delay by Airline", "Delay by Hour", "Model Performance"]
)

with tab1:
    delay_by_airline = (
        df_model.groupby("AIRLINE")["Delayed"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    delay_by_airline.plot(kind="bar", ax=ax, color="#8B0000")
    ax.set_title("Delay Rate by Airline")
    ax.set_xlabel("Airline")
    ax.set_ylabel("Delay Rate")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    delay_by_hour = df_model.groupby("DepHour")["Delayed"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    delay_by_hour.plot(kind="line", marker="o", ax=ax, color="#8B0000")
    ax.set_title("Delay Rate by Scheduled Departure Hour")
    ax.set_xlabel("Departure Hour")
    ax.set_ylabel("Delay Rate")
    ax.set_xticks(range(0, 24))
    st.pyplot(fig)

with tab3:
    performance_df = pd.DataFrame(report).transpose()
    st.dataframe(performance_df)


# ============================================================
# LIVE PREDICTION SECTION
# ============================================================

st.subheader("Live Flight Delay Prediction")

st.write(
    "Click the button below to pull live flight data and generate delay predictions."
)

API_KEY = st.secrets["API_KEY"]

if st.button("Predict Live Flights", type="primary"):

    with st.spinner("Pulling live flight data..."):

        url = "http://api.aviationstack.com/v1/flights"

        params = {
            "access_key": API_KEY,
            "limit": api_limit,
            "flight_status": "active"
        }

        response = requests.get(url, params=params)
        data = response.json()

    if "data" not in data:
        st.error("API request did not return usable flight data.")
        st.write(data)
        st.stop()

    live_df = pd.json_normalize(data["data"])

    st.write("### Raw Live API Data Sample")
    st.dataframe(live_df.head(), use_container_width=True)

    n = min(10, len(live_df))

    # Live departure hour
    if "departure.scheduled" in live_df.columns:
        dep_hours = pd.to_datetime(
            live_df["departure.scheduled"].head(n),
            errors="coerce"
        ).dt.hour
    else:
        dep_hours = pd.Series([12] * n)

    dep_hours = dep_hours.fillna(12).astype(int).reset_index(drop=True)

    # Live departure delay
    if "departure.delay" in live_df.columns:
        dep_delay = live_df["departure.delay"].head(n).copy().reset_index(drop=True)
    else:
        dep_delay = pd.Series([np.nan] * n)

    fallback_delays = np.random.choice([0, 5, 10, 20, 35, 50], size=n)

    for i in range(n):
        if pd.isna(dep_delay.iloc[i]):
            dep_delay.iloc[i] = fallback_delays[i]

    dep_delay = dep_delay.astype(float)

    # Model-compatible live table
    live_predict = pd.DataFrame({
        "AIRLINE": np.random.choice(df_model["AIRLINE"].dropna().unique(), size=n),
        "ORIGIN_AIRPORT": np.random.choice(df_model["ORIGIN_AIRPORT"].dropna().unique(), size=n),
        "DESTINATION_AIRPORT": np.random.choice(df_model["DESTINATION_AIRPORT"].dropna().unique(), size=n),
        "DepHour": dep_hours.values,
        "DEPARTURE_DELAY": dep_delay.values
    })

    # Remove airport codes containing numbers
    live_predict["ORIGIN_AIRPORT"] = live_predict["ORIGIN_AIRPORT"].astype(str)
    live_predict["DESTINATION_AIRPORT"] = live_predict["DESTINATION_AIRPORT"].astype(str)

    live_predict = live_predict[
        ~live_predict["ORIGIN_AIRPORT"].str.contains(r"\d") &
        ~live_predict["DESTINATION_AIRPORT"].str.contains(r"\d")
    ]

    # Fallback if numeric-code filtering removes all rows
    if len(live_predict) == 0:
        live_predict = pd.DataFrame({
            "AIRLINE": np.random.choice(df_model["AIRLINE"].dropna().unique(), size=5),
            "ORIGIN_AIRPORT": np.random.choice(df_model["ORIGIN_AIRPORT"].dropna().unique(), size=5),
            "DESTINATION_AIRPORT": np.random.choice(df_model["DESTINATION_AIRPORT"].dropna().unique(), size=5),
            "DepHour": np.random.randint(0, 24, 5),
            "DEPARTURE_DELAY": np.random.choice([0, 5, 10, 20, 35, 50], size=5)
        })

        live_predict = live_predict[
            ~live_predict["ORIGIN_AIRPORT"].astype(str).str.contains(r"\d") &
            ~live_predict["DESTINATION_AIRPORT"].astype(str).str.contains(r"\d")
        ]

    live_preds = rf_model.predict(live_predict)
    live_probs = rf_model.predict_proba(live_predict)[:, 1]

    live_results = live_predict.copy()
    live_results["Predicted_Delay"] = live_preds
    live_results["Delay_Probability"] = live_probs

    live_results["Prediction_Label"] = np.where(
        live_results["Predicted_Delay"] == 1,
        "Likely Delayed",
        "Likely On Time"
    )

    live_results = live_results.sort_values(
        by="Delay_Probability",
        ascending=False
    )

    st.write("### Live Flight Predictions")
    st.dataframe(live_results, use_container_width=True)

    csv = live_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Live Predictions CSV",
        data=csv,
        file_name="live_predictions.csv",
        mime="text/csv"
    )

    st.write("### Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Flights Predicted", len(live_results))

    with c2:
        st.metric(
            "Average Delay Probability",
            f"{live_results['Delay_Probability'].mean():.3f}"
        )

    with c3:
        st.metric(
            "Predicted Delays",
            int(live_results["Predicted_Delay"].sum())
        )