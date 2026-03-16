# Geospatial-flood-risk-prediction

Project Overview

This repository contains a machine learning application designed to predict flood risks using geospatial data. The system utilizes an XGBoost classifier to analyze environmental variables and provides an interactive interface for spatial visualization.

Technical Stack

Framework: Streamlit
Model: XGBoost (Gradient Boosting)
Spatial Data: Folium

Project Structure

The project is organized to separate the user interface from the core prediction logic:
/app: Contains the Streamlit entry point and UI components.
/src: Contains the predict.py module and model inference logic.
/requirements.txt: Managed dependencies for deployment.

Implementation Details

The model processes geospatial inputs to calculate risk levels. By integrating folium, the application allows users to interact with geographical coordinates and receive real-time predictions based on historical and environmental features.

Deployment Link: https://geospatial-flood-risk-predictor.streamlit.app/
