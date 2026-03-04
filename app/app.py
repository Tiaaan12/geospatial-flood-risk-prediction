

import streamlit as st
import folium
from streamlit_folium import st_folium
from src.predict import predict_single


st.set_page_config(page_title="Flood Risk Predictor", layout="wide")


st.title("🌊 Geospatial Flood Risk Predictor")