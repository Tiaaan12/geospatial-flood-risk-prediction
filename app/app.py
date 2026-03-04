import streamlit as st
import folium
from streamlit_folium import st_folium
from src.predict import predict_single

st.set_page_config(page_title="Flood Risk Predictor", layout="wide")

st.title("Geospatial Flood Risk Predictor")

st.sidebar.header("Input Parameters")
latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 0.0)
longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 0.0)
duration = st.sidebar.number_input("Duration (days)", 0.0, 100.0, 1.0)
rainfall = st.sidebar.number_input("Rainfall", 0.0, 1000.0, 50.0)
elevation = st.sidebar.number_input("Elevation", 0.0, 5000.0, 100.0)
slope = st.sidebar.number_input("Slope", 0.0, 90.0, 5.0)

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.probability = None
    st.session_state.lat = None
    st.session_state.lon = None
    
if st.sidebar.button("Predict Flood Risk"):

    prediction, probability = predict_single(
        latitude,
        longitude,
        duration,
        rainfall,
        elevation,
        slope
    )

    st.session_state.prediction = prediction
    st.session_state.probability = probability
    st.session_state.lat = latitude
    st.session_state.lon = longitude
