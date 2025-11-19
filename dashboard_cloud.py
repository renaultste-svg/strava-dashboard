import base64
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

import requests
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components  # NEW

# --- D'abord : config de la page (sans markdown) ---

icon_path = Path("apple-touch-icon.png")
page_icon = "apple-touch-icon.png" if icon_path.exists() else None

st.set_page_config(
    page_title="Dashboard Strava – Miguel",
    page_icon=page_icon,  # notre icône si trouvée
    layout="wide",
)

# --- Ensuite seulement : liens pour iPhone / favicon ---

if icon_path.exists():
    st.markdown(
        """
        <link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
        <link rel="icon" type="image/png" href="apple-touch-icon.png">
        """,
        unsafe_allow_html=True,
    )

# =========================
#  CONFIG & GESTION TOKENS
# =========================

BASE_URL = "https://www.strava.com/api/v3"
TOKENS_FILE = Path("tokens_cloud.json")

CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
INITIAL_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]

DEFAULT_WEIGHT_KG = float(st.secrets.get("ATHLETE_WEIGHT_KG", 89.0))

# NEW – constantes pour les équivalences gourmandes
CAL_PER_PIZZA_SLICE = 250.0_
