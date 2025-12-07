# app.py
"""
Air Quality Predictive Dashboard (final)
Features:
 - Map (folium) click -> fill coords
 - City geocode (geopy) or HTTP fallback
 - Train ML models (7 models) on pollutant_min/avg/max -> AQI_Class
 - Live prediction using chosen model or rule-based
 - AQI color gauge (Plotly)
 - Live AQI fetch from OpenAQ (city-wise) with fallback
 - Alert banners when AQI is Unhealthy+
 - Trend charts for session predictions
 - Dark mode toggle (CSS)
 - No PDF export (removed)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# Optional libs (graceful)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM = True
except Exception:
    FOLIUM = False

try:
    from geopy.geocoders import Nominatim
    GEOPY = True
except Exception:
    GEOPY = False

try:
    import requests
    REQUESTS = True
except Exception:
    REQUESTS = False

# scikit-learn (required for ML)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN = True
except Exception:
    SKLEARN = False

# -------------------------
# Page config + dark-mode toggle
# -------------------------
st.set_page_config(page_title="Air Quality — Smart Dashboard", layout="wide", page_icon="🌫️")
st.title("🌫️ Air Quality — Smart Dashboard (AQI meter, map, alerts, trends)")

# Dark mode toggle
dark = st.sidebar.checkbox("Enable dark mode (UI)", value=False)
if dark:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e6eef8; }
        .css-1d391kg { color: #e6eef8; }
        .st-bf { color: #e6eef8; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Session state for history
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # stores prediction dicts

# -------------------------
# Helper functions
# -------------------------
def categorize_aqi_numeric(aqi):
    if pd.isna(aqi):
        return "Unknown"
    aqi = float(aqi)
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy_SG"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "VeryUnhealthy"
    return "Hazardous"

def aqi_category_and_value_from_avg(avg):
    avg = float(avg)
    if avg <= 50: return "Good", 25
    if avg <= 100: return "Moderate", 75
    if avg <= 150: return "Unhealthy for Sensitive Groups", 125
    if avg <= 200: return "Unhealthy", 175
    if avg <= 300: return "Very Unhealthy", 250
    return "Hazardous", 400

def eco_suggestions(category):
    c = category.lower()
    if "hazard" in c or "very" in c or ("unhealthy" in c and "sensitive" not in c):
        return [
            "Avoid outdoor exercise; prefer indoor activities.",
            "Use N95/FFP2 masks if outside.",
            "Reduce vehicle usage; encourage public transport.",
            "Local authorities: consider temporary restrictions."
        ]
    if "sensitive" in c:
        return [
            "Sensitive groups: limit prolonged outdoor exposure.",
            "Keep inhalers/meds handy.",
            "Monitor symptoms and seek care if needed."
        ]
    if "moderate" in c:
        return [
            "Prefer public transport or carpool.",
            "Reduce open burning and idling vehicles.",
            "Check daily AQI before long outdoor activity."
        ]
    return [
        "Air is good — promote walking & cycling.",
        "Support local tree-planting and clean-air drives."
    ]

def geocode_city_http(city_name):
    if not REQUESTS: 
        return None, None
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format":"json", "limit":1}
        headers = {"User-Agent":"aqi_dashboard"}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        j = r.json()
        if not j: return None, None
        return float(j[0]["lat"]), float(j[0]["lon"])
    except Exception:
        return None, None

def geocode_city(city_name):
    if GEOPY:
        try:
            geolocator = Nominatim(user_agent="aqi_app_streamlit")
            loc = geolocator.geocode(city_name, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
    return geocode_city_http(city_name)

def fetch_openaq_pm25(city_name):
    # returns numeric pm25 value or None
    if not REQUESTS:
        return None
    try:
        url = f"https://api.openaq.org/v2/latest?city={city_name}&parameter=pm25"
        r = requests.get(url, timeout=8)
        j = r.json()
        vals = []
        for res in j.get("results", [])[:12]:
            for m in res.get("measurements", []):
                if m.get("parameter") == "pm25" and m.get("value") is not None:
                    vals.append(m["value"])
        if vals:
            return float(np.nanmean(vals))
    except Exception:
        pass
    return None

def get_aqi_color(aqi):
    # returns hex color for gauge and banner
    if aqi <=50: return "#2ecc71"   # green
    if aqi <=100: return "#f1c40f"  # yellow
    if aqi <=150: return "#e67e22"  # orange
    if aqi <=200: return "#e74c3c"  # red
    if aqi <=300: return "#8e44ad"  # purple
    return "#6b1e4a"                # maroon

# -------------------------
# Sidebar: Upload or demo
# -------------------------
st.sidebar.header("Dataset & Options")
uploaded = st.sidebar.file_uploader("Upload cleaned CSV (optional)", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo sample if no upload", value=True)

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("CSV loaded")
else:
    if use_demo:
        rng = np.random.RandomState(42)
        n = 300
        pollutant_avg = np.clip(rng.normal(80, 60, size=n), 5, 500)
        df = pd.DataFrame({
            "country": ["India"]*n,
            "state": ["State"]*n,
            "city": rng.choice(["Delhi","Mumbai","Bengaluru","Kolkata","Chennai"], size=n),
            "station": rng.choice(["S1","S2","S3"], size=n),
            "last_update": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="H"),
            "latitude": rng.uniform(8,30,size=n),
            "longitude": rng.uniform(68,88,size=n),
            "pollutant_min": np.clip(pollutant_avg - rng.uniform(5,40,size=n),0,None),
            "pollutant_max": pollutant_avg + rng.uniform(1,60,size=n),
            "pollutant_avg": pollutant_avg
        })
    else:
        st.info("Upload CSV or enable demo sample in sidebar.")
        st.stop()

# Ensure required columns exist
required = ["pollutant_min","pollutant_avg","pollutant_max"]
for c in required:
    if c not in df.columns:
        st.error(f"Dataset missing required column: {c}")
        st.stop()

# Create AQI_Class if missing (approx from pollutant_avg)
if "AQI_Class" not in df.columns:
    st.info("AQI_Class not found — estimating from pollutant_avg (approx).")
    df["AQI_Estimated"] = df["pollutant_avg"]
    df["AQI_Class"] = df["AQI_Estimated"].apply(categorize_aqi_numeric)
else:
    if "AQI_Estimated" not in df.columns:
        df["AQI_Estimated"] = df["pollutant_avg"]

# Show data preview
st.subheader("Dataset preview")
st.dataframe(df.head(6))
st.write("Columns:", list(df.columns))

# -------------------------
# Feature set & model train
# -------------------------
st.markdown("---")
st.subheader("Model training (pollutant_min/avg/max)")

# choose extra features optionally
available_extra = [c for c in ["latitude","longitude","city","station","state"] if c in df.columns]
include_extras = st.multiselect("Include extra features", options=available_extra, default=["latitude","longitude"] if "latitude" in available_extra else [])

X = df[["pollutant_min","pollutant_avg","pollutant_max"] + include_extras].copy()
# simple encoding for object columns
for c in X.select_dtypes(include="object").columns:
    X[c], _ = pd.factorize(X[c].astype(str))

y = df["AQI_Class"].astype(str)

# model definitions
if SKLEARN:
    model_defs = {
        "KNN": KNeighborsClassifier(),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network": MLPClassifier(max_iter=800, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
else:
    model_defs = {}
    st.warning("scikit-learn not available — ML training disabled.")

# safe train/test split (disable stratify if rare classes)
if SKLEARN:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    counts = pd.Series(y_enc).value_counts()
    if counts.min() < 2:
        strat = None
        st.warning("Some classes have <2 samples — stratify disabled.")
    else:
        strat = y_enc

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=strat)

    # scale numeric columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    st.session_state["scaler"] = scaler
    st.session_state["X_columns"] = X.columns.tolist()

    # Train all models and show evaluation
    eval_results = []
    trained = {}
    for name, mdl in model_defs.items():
        try:
            mdl.fit(X_train_s, y_train)
            y_pred = mdl.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            try:
                prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
                rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            except Exception:
                prec = rec = f1 = None
            eval_results.append({"Model":name,"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1})
            trained[name] = mdl
        except Exception as e:
            st.error(f"Training failed for {name}: {e}")

    eval_df = pd.DataFrame(eval_results).sort_values("F1", ascending=False)
    st.write("Model evaluation")
    st.dataframe(eval_df)

    st.session_state["trained_models"] = trained
    if not eval_df.empty:
        st.success(f"Best model by F1: {eval_df.iloc[0]['Model']} (F1={eval_df.iloc[0]['F1']:.3f})")
else:
    st.info("Install scikit-learn to enable ML training.")

# -------------------------
# Live prediction + AQI meter + alerts + trends
# -------------------------
st.markdown("---")
st.subheader("Live prediction + AQI meter + alerts + trends")

# Inputs and map columns
col_left, col_right = st.columns([1,1])

with col_left:
    pmin = st.number_input("Min pollutant value", value=float(X["pollutant_min"].median()))
    pavg = st.number_input("Avg pollutant value", value=float(X["pollutant_avg"].median()))
    pmax = st.number_input("Max pollutant value", value=float(X["pollutant_max"].median()))
    city_input = st.text_input("City (optional)", value="")
    lat_input = st.number_input("Latitude (optional)", value=0.0, format="%.6f", key="lat_input_main")
    lon_input = st.number_input("Longitude (optional)", value=0.0, format="%.6f", key="lon_input_main")

    # model choices
    model_choices = ["Rule-based (always available)"]
    if SKLEARN and st.session_state.get("trained_models"):
        model_choices += list(st.session_state["trained_models"].keys())

    chosen_model = st.selectbox("Choose model for prediction", model_choices)

    if st.button("Predict & Log"):
        # do prediction
        if chosen_model != "Rule-based (always available)" and SKLEARN:
            cols = st.session_state.get("X_columns", X.columns.tolist())
            x_in = {}
            for c in cols:
                if c == "pollutant_min": x_in[c] = pmin
                elif c == "pollutant_max": x_in[c] = pmax
                elif c == "pollutant_avg": x_in[c] = pavg
                elif c == "latitude": x_in[c] = lat_input
                elif c == "longitude": x_in[c] = lon_input
                else: x_in[c] = 0
            Xsamp = pd.DataFrame([x_in])[cols]
            try:
                Xs = st.session_state["scaler"].transform(Xsamp)
                mdl = st.session_state["trained_models"][chosen_model]
                pred_enc = mdl.predict(Xs)[0]
                pred_label = le.inverse_transform([pred_enc])[0]
                pred_aqi = float(Xsamp["pollutant_avg"].iloc[0]) if "pollutant_avg" in Xsamp.columns else pavg
                rec = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": chosen_model,
                    "pred_label": pred_label,
                    "aqi": pred_aqi,
                    "min": pmin, "avg": pavg, "max": pmax,
                    "city": city_input, "lat": lat_input, "lon": lon_input
                }
                st.session_state.history.append(rec)
                st.success(f"{chosen_model} → {pred_label} (AQI≈{pred_aqi:.1f})")
            except Exception as e:
                st.error(f"ML prediction failed: {e}")
        else:
            pred_label, pred_aqi_val = aqi_category_and_value_from_avg(pavg)
            rec = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": "Rule-based",
                "pred_label": pred_label,
                "aqi": pred_aqi_val,
                "min": pmin, "avg": pavg, "max": pmax,
                "city": city_input, "lat": lat_input, "lon": lon_input
            }
            st.session_state.history.append(rec)
            st.success(f"Rule-based → {pred_label} (AQI {pred_aqi_val})")

with col_right:
    # AQI gauge & alert
    st.markdown("### AQI meter & latest prediction")
    if st.session_state.history:
        last = st.session_state.history[-1]
        aqi_val = float(last["aqi"])
        label = last["pred_label"]
        color = get_aqi_color(aqi_val)

        if PLOTLY:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=aqi_val,
                title={"text": f"Latest: {label}"},
                gauge={
                    "axis": {"range":[0,500]},
                    "bar": {"color": color},
                    "steps":[
                        {"range":[0,50],"color":"#2ecc71"},
                        {"range":[50,100],"color":"#f1c40f"},
                        {"range":[100,150],"color":"#e67e22"},
                        {"range":[150,200],"color":"#e74c3c"},
                        {"range":[200,300],"color":"#8e44ad"},
                        {"range":[300,500],"color":"#6b1e4a"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"AQI ≈ {aqi_val} — {label}")

        # Alert banners for unhealthy readings
        if aqi_val > 300:
            st.error("🚨 Emergency: AQI in Hazardous range — avoid outdoor exposure.")
        elif aqi_val > 200:
            st.error("⚠️ Very Unhealthy: high health risk for all groups.")
        elif aqi_val > 150:
            st.warning("⚠ Unhealthy: sensitive groups should avoid outdoor exertion.")
        elif aqi_val > 100:
            st.info("Notice: Unhealthy for sensitive groups.")
        else:
            st.success("Air quality is acceptable / moderate.")

        # show suggestions
        st.markdown("**Eco suggestions:**")
        for s in eco_suggestions(label):
            st.write("•", s)
    else:
        st.info("No predictions yet. Use inputs and click 'Predict & Log'.")

# -------------------------
# Trend charts for session predictions
# -------------------------
st.markdown("---")
st.subheader("Trends (session history)")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df["time_dt"] = pd.to_datetime(hist_df["time"])
    hist_df_sorted = hist_df.sort_values("time_dt")

    # line chart of AQI over time
    if PLOTLY:
        fig_trend = px.line(hist_df_sorted, x="time_dt", y="aqi", color="model", markers=True, title="Predicted AQI over session")
        st.plotly_chart(fig_trend, use_container_width=True)
        # histogram of categories
        fig_hist = px.histogram(hist_df_sorted, x="pred_label", title="Predicted AQI categories (session)")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.line_chart(hist_df_sorted.set_index("time_dt")["aqi"])
        st.bar_chart(hist_df_sorted["pred_label"].value_counts())
else:
    st.info("No history to show trends. Log some predictions first.")

# -------------------------
# Map (interactive) - clicking fills latitude/longitude and triggers auto-predict flag
# -------------------------
st.markdown("---")
st.subheader("Interactive Map (click to pick coordinates)")

map_center = [20.5937, 78.9629]
zoom_start = 5
if FOLIUM:
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.LatLngPopup().add_to(m)
    map_result = st_folium(m, width=900, height=450)
    if map_result and map_result.get("last_clicked"):
        lc = map_result["last_clicked"]
        latc, lonc = lc["lat"], lc["lng"]
        st.success(f"Map click coords: {latc:.6f}, {lonc:.6f}")
        # Fill main numeric inputs via session_state keys (user may need to press Predict)
        st.session_state["lat_input_main"] = float(latc)
        st.session_state["lon_input_main"] = float(lonc)
        # also update local inputs shown on screen
        st.experimental_set_query_params()  # no-op used to force small state sync in some environments
else:
    st.info("Map not available — install streamlit-folium & folium")

# -------------------------
# Live AQI API fetch (OpenAQ) for a city
# -------------------------
st.markdown("---")
st.subheader("Live AQI (OpenAQ) — fetch recent PM2.5 for a city")

city_query = st.text_input("City to fetch OpenAQ PM2.5 (e.g., Delhi)", value="")
if st.button("Fetch Live PM2.5"):
    pm25 = fetch_openaq_pm25(city_query)
    if pm25 is not None:
        st.success(f"OpenAQ PM2.5 (avg of stations): {pm25:.2f} µg/m³")
        cat = categorize_aqi_numeric(pm25)
        st.info(f"Approx category from PM2.5: {cat}")
        if PLOTLY:
            fig_live = go.Figure(go.Indicator(mode="gauge+number", value=pm25,
                                             gauge={"axis":{"range":[0,500]},
                                                    "bar":{"color":get_aqi_color(pm25)}}))
            st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.warning("Could not fetch live data or no PM2.5 found — showing simulated sample")
        sim = max(5, np.random.normal(70, 35))
        st.info(f"Simulated PM2.5: {sim:.1f} µg/m³ → {categorize_aqi_numeric(sim)}")

# -------------------------
# Explainability + model inspect
# -------------------------
st.markdown("---")
st.subheader("Model inspect & explainability (Random Forest importance)")

if SKLEARN and st.session_state.get("trained_models") and "Random Forest" in st.session_state["trained_models"]:
    rf = st.session_state["trained_models"]["Random Forest"]
    if hasattr(rf, "feature_importances_"):
        feat_names = st.session_state.get("X_columns", X.columns.tolist())
        fi = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
        st.dataframe(fi)
        if PLOTLY:
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature importances")
            st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Train models to inspect feature importances.")

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.subheader("Notes")
st.markdown("""
- AQI is approximated from `pollutant_avg` in this app when full pollutant-specific AQI formulas are not available.
- For production-grade AQI use pollutant-specific breakpoints (CPCB/EPA) or serve a trained model via an API.
- Optional packages (folium, streamlit-folium, geopy, plotly, requests) improve UX and live data.
""")

# End of app
