import base64
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

import requests
import pandas as pd
import streamlit as st
import altair as alt

# --- Ajouter l’icône iPhone + favicon (apple-touch-icon) ---
icon_path = Path("apple-touch-icon.png")
page_icon = None

if icon_path.exists():
    with icon_path.open("rb") as f:
        b64_icon = base64.b64encode(f.read()).decode("utf-8")

    # Icône pour iPhone (écran d’accueil)
    st.markdown(
        f"""
        <link rel="apple-touch-icon" sizes="180x180"
              href="data:image/png;base64,{b64_icon}">
        <link rel="icon" type="image/png"
              href="data:image/png;base64,{b64_icon}">
        """,
        unsafe_allow_html=True,
    )

    # Pour Streamlit (favicon)
    page_icon = "apple-touch-icon.png"


# =========================
#  CONFIG & GESTION TOKENS
# =========================

BASE_URL = "https://www.strava.com/api/v3"
TOKENS_FILE = Path("tokens_cloud.json")

CLIENT_ID = st.secrets["STRAVA_CLIENT_ID"]
CLIENT_SECRET = st.secrets["STRAVA_CLIENT_SECRET"]
INITIAL_REFRESH_TOKEN = st.secrets["STRAVA_REFRESH_TOKEN"]

DEFAULT_WEIGHT_KG = float(st.secrets.get("ATHLETE_WEIGHT_KG", 89.0))


def save_tokens(tokens: dict) -> None:
    """Sauvegarde les tokens dans un fichier local du cloud."""
    with TOKENS_FILE.open("w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2)


def load_tokens_from_file() -> dict | None:
    """Charge les tokens depuis le fichier local (si existe)."""
    if not TOKENS_FILE.exists():
        return None
    with TOKENS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def refresh_tokens(refresh_token: str) -> dict:
    """Appelle Strava pour obtenir un access_token + refresh_token frais."""
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    resp.raise_for_status()
    tokens = resp.json()
    save_tokens(tokens)
    return tokens


def get_valid_tokens() -> dict:
    """
    1) Essaie de lire tokens_cloud.json
    2) Si absent, utilise le refresh_token stocké dans les secrets
    3) Si expiré, appelle Strava pour en générer de nouveaux
    """
    tokens = load_tokens_from_file()

    if tokens is None:
        # Premier lancement cloud : on part du refresh_token des secrets
        tokens = refresh_tokens(INITIAL_REFRESH_TOKEN)

    now = int(time.time())
    expires_at = tokens.get("expires_at", 0)

    # S'il reste moins d'une minute de validité, on rafraîchit
    if expires_at - now < 60:
        tokens = refresh_tokens(tokens["refresh_token"])

    return tokens


def get_athlete_weight(tokens: dict) -> float:
    athlete = tokens.get("athlete") or {}
    w = athlete.get("weight")
    if isinstance(w, (int, float)) and w > 0:
        return float(w)
    return DEFAULT_WEIGHT_KG


# =========================
#  RÉCUPÉRATION ACTIVITÉS
# =========================

def fetch_recent_activities(access_token: str, days: int = 90) -> list[dict]:
    """Récupère les activités des X derniers jours."""
    headers = {"Authorization": f"Bearer {access_token}"}
    after_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    page = 1
    per_page = 200
    all_activities: list[dict] = []

    with st.spinner(f"Récupération des {days} derniers jours d'activités…"):
        while True:
            params = {"after": after_ts, "page": page, "per_page": per_page}
            resp = requests.get(
                f"{BASE_URL}/athlete/activities",
                headers=headers,
                params=params,
                timeout=20,
            )
            resp.raise_for_status()
            activities = resp.json()
            if not activities:
                break
            all_activities.extend(activities)
            page += 1

    return all_activities


def activities_to_dataframe(
    activities: list[dict],
    athlete_weight_kg: float,
) -> pd.DataFrame:
    """Transforme la liste JSON en DataFrame exploitable."""
    rows = []

    for a in activities:
        sport = a.get("sport_type") or a.get("type")
        distance_m = a.get("distance") or 0.0
        moving_time_s = a.get("moving_time") or 0
        start_date = a.get("start_date_local") or a.get("start_date")

        distance_km = distance_m / 1000
        moving_time_min = moving_time_s / 60

        calories = a.get("calories")

        # Vélo : conversion kJ -> kcal si besoin
        if calories is None and a.get("kilojoules") is not None:
            calories = a["kilojoules"] * 0.239

        # CAP sans calories : 1 kcal / kg / km
        if calories is None and sport in ("Run", "TrailRun", "NordicSki"):
            calories = athlete_weight_kg * distance_km

        rows.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "sport": sport,
                "distance_km": distance_km,
                "moving_time_min": moving_time_min,
                "start_date": start_date,
                "calories": calories or 0.0,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    iso = df["start_date"].dt.isocalendar()
    df["iso_year"] = iso.year
    df["iso_week"] = iso.week
    df["week_label"] = (
        df["iso_year"].astype(str)
        + "-W"
        + df["iso_week"].astype(str).astype(str).str.zfill(2)
    )

    return df


# =========================
#  ANALYSE 7J / 30J & ACWR
# =========================

def _format_pace(avg_min_per_km: float | None) -> str:
    if avg_min_per_km is None:
        return "—"
    mins = int(avg_min_per_km)
    secs = int(round((avg_min_per_km - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins}:{secs:02d}/km"


def summarize_period(df: pd.DataFrame, days: int, last_date: datetime) -> dict:
    """Résumé pour une période (tous sports + CAP uniquement)."""
    if df.empty:
        return {
            "total_kcal": 0.0,
            "n_activities": 0,
            "total_hours": 0.0,
            "run_sessions": 0,
            "run_distance": 0.0,
            "run_pace": None,
        }

    start = last_date - timedelta(days=days - 1)
    mask = (df["start_date"] >= start) & (df["start_date"] <= last_date)
    sub = df.loc[mask]

    if sub.empty:
        return {
            "total_kcal": 0.0,
            "n_activities": 0,
            "total_hours": 0.0,
            "run_sessions": 0,
            "run_distance": 0.0,
            "run_pace": None,
        }

    total_kcal = float(sub["calories"].sum())
    n_activities = int(sub.shape[0])
    total_hours = float(sub["moving_time_min"].sum() / 60)

    run = sub[sub["sport"].isin(["Run", "TrailRun"])]
    run_sessions = int(run.shape[0])
    run_distance = float(run["distance_km"].sum())

    if run_distance > 0:
        avg_pace = (run["moving_time_min"].sum() / run_distance)
    else:
        avg_pace = None

    return {
        "total_kcal": total_kcal,
        "n_activities": n_activities,
        "total_hours": total_hours,
        "run_sessions": run_sessions,
        "run_distance": run_distance,
        "run_pace": avg_pace,
    }


def compute_acwr(df: pd.DataFrame, last_date: datetime) -> dict:
    """
    Calcule la variation de volume CAP :
    - Distance CAP des 7 derniers jours
    - vs les 7 jours précédents
    """
    if df.empty:
        return {
            "current": 0.0,
            "previous": 0.0,
            "ratio": None,
            "delta_pct": None,
        }

    current_start = last_date - timedelta(days=6)
    prev_start = current_start - timedelta(days=7)
    prev_end = current_start - timedelta(seconds=1)

    run_df = df[df["sport"].isin(["Run", "TrailRun"])]

    mask_current = (run_df["start_date"] >= current_start) & (
        run_df["start_date"] <= last_date
    )
    mask_prev = (run_df["start_date"] >= prev_start) & (
        run_df["start_date"] <= prev_end
    )

    d_current = float(run_df.loc[mask_current, "distance_km"].sum())
    d_prev = float(run_df.loc[mask_prev, "distance_km"].sum())

    if d_prev > 0:
        ratio = d_current / d_prev
        delta_pct = (d_current - d_prev) / d_prev * 100
    else:
        ratio = None
        delta_pct = None

    return {
        "current": d_current,
        "previous": d_prev,
        "ratio": ratio,
        "delta_pct": delta_pct,
    }


def build_text_report(
    df: pd.DataFrame,
    summary_7: dict,
    summary_30: dict,
    acwr: dict,
) -> str:
    """Texte à copier-coller pour Miguel (toi + moi)."""
    if df.empty:
        return "Aucune activité trouvée sur la période."

    last_date = df["start_date"].max().date()

    def fmt_hours(h: float) -> str:
        return f"{h:.1f} h"

    def fmt_kcal(v: float) -> str:
        return f"{int(round(v))} kcal"

    pace_7 = _format_pace(summary_7["run_pace"])
    pace_30 = _format_pace(summary_30["run_pace"])

    d7 = acwr["current"]
    d7_prev = acwr["previous"]
    delta_pct = acwr["delta_pct"]

    if delta_pct is None:
        delta_str = "n.c."
    else:
        sign = "+" if delta_pct >= 0 else ""
        delta_str = f"{sign}{delta_pct:.1f} %"

    report = f"""APPORT D’ENTRAÎNEMENT – EXTRACTION STRAVA
==========================================
Date de référence (dernière activité) : {last_date}

Périodes analysées :
  - 7 derniers jours (semaine glissante)
  - 30 derniers jours

📆 7 DERNIERS JOURS
------------------
Total calories           : {fmt_kcal(summary_7["total_kcal"])}
Nombre d’activités       : {summary_7["n_activities"]}
Temps total              : {fmt_hours(summary_7["total_hours"])}

Course à pied :
  - Séances             : {summary_7["run_sessions"]}
  - Distance totale     : {summary_7["run_distance"]:.2f} km
  - Allure moyenne CAP  : {pace_7}

📆 30 DERNIERS JOURS
-------------------
Total calories           : {fmt_kcal(summary_30["total_kcal"])}
Nombre d’activités       : {summary_30["n_activities"]}
Temps total              : {fmt_hours(summary_30["total_hours"])}

Course à pied :
  - Séances             : {summary_30["run_sessions"]}
  - Distance totale     : {summary_30["run_distance"]:.2f} km
  - Allure moyenne CAP  : {pace_30}

Variation volume CAP 7j vs 7j précédents
-----------------------------------------
Distance CAP 7j en cours      : {d7:.2f} km
Distance CAP 7j précédents    : {d7_prev:.2f} km
Évolution                     : {delta_str}

Miguel-ready summary
---------------------
- CAP 7j : {summary_7["run_distance"]:.2f} km sur {summary_7["run_sessions"]} séances.
- Allure moyenne CAP (7j) : {pace_7}
- Calories totales 7j (tous sports) : {fmt_kcal(summary_7["total_kcal"])}
- CAP 30j : {summary_30["run_distance"]:.2f} km sur {summary_30["run_sessions"]} séances.
- Calories CAP 30j : {fmt_kcal(summary_30["total_kcal"])}
- Variation volume CAP 7j vs 7j précédents : {delta_str}
"""

    return report


# =========================
#  DASHBOARD STREAMLIT
# =========================

st.set_page_config(
    page_title="Dashboard Strava – Miguel",
    page_icon=page_icon,  # notre icône si trouvée
    layout="wide",
)


st.title("📊 Dashboard Strava – Mode Miguel Ultra Complet")

st.markdown(
    """
Ce tableau de bord est **perso** : il sert à suivre ton volume CAP + vélo,
et à générer un petit rapport à coller dans la discussion avec Miguel.
"""
)

# ----- Chargement des données -----

try:
    tokens = get_valid_tokens()
    access_token = tokens["access_token"]
    athlete_weight = get_athlete_weight(tokens)

    st.sidebar.header("⚙️ Options")
    days_range = st.sidebar.slider(
        "Fenêtre d'analyse principale",
        min_value=30,
        max_value=180,
        value=90,
        step=30,
        help="Nombre de jours utilisés pour les graphiques hebdos.",
    )

    activities = fetch_recent_activities(access_token, days=days_range)
    df = activities_to_dataframe(activities, athlete_weight)

except Exception as e:
    st.error(f"Erreur lors de la connexion à Strava : {e}")
    st.stop()

if df.empty:
    st.warning("Aucune activité trouvée sur la période choisie.")
    st.stop()

last_date = df["start_date"].max()

st.markdown(
    f"**Dernière activité Strava détectée :** {last_date.strftime('%Y-%m-%d %H:%M')}"
)
st.markdown(f"**Poids utilisé pour les calculs CAP :** {athlete_weight:.1f} kg")

# ----- Résumés 7j / 30j + ACWR -----

summary_7 = summarize_period(df, days=7, last_date=last_date)
summary_30 = summarize_period(df, days=30, last_date=last_date)
acwr = compute_acwr(df, last_date=last_date)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📆 7 derniers jours")
    st.metric(
        "Calories totales",
        f"{int(round(summary_7['total_kcal']))} kcal",
    )
    st.metric(
        "Temps total",
        f"{summary_7['total_hours']:.1f} h",
    )
    st.metric(
        "CAP – distance",
        f"{summary_7['run_distance']:.2f} km ({summary_7['run_sessions']} séances)",
    )
    st.metric(
        "Allure CAP moyenne",
        _format_pace(summary_7["run_pace"]),
    )

with col2:
    st.subheader("📆 30 derniers jours")
    st.metric(
        "Calories totales",
        f"{int(round(summary_30['total_kcal']))} kcal",
    )
    st.metric(
        "Temps total",
        f"{summary_30['total_hours']:.1f} h",
    )
    st.metric(
        "CAP – distance",
        f"{summary_30['run_distance']:.2f} km ({summary_30['run_sessions']} séances)",
    )
    st.metric(
        "Allure CAP moyenne",
        _format_pace(summary_30["run_pace"]),
    )

with col3:
    st.subheader("📈 Charge CAP 7j")
    d_cur = acwr["current"]
    d_prev = acwr["previous"]
    ratio = acwr["ratio"]
    delta_pct = acwr["delta_pct"]

    if ratio is None:
        st.metric("ACWR (7j / 7j-1)", "n.c.")
        st.metric("Δ distance 7j vs 7j-1", "n.c.")
    else:
        st.metric("ACWR (7j / 7j-1)", f"{ratio:.2f}")
        sign = "+" if delta_pct >= 0 else ""
        st.metric("Δ distance 7j vs 7j-1", f"{sign}{delta_pct:.1f} %")

    st.caption(
        "ACWR ~ ratio de volume CAP : 0.8–1.3 = zone safe, >1.5 = augmentation agressive."
    )

# ----- Graphiques hebdo -----

st.markdown("---")
st.subheader("📊 Charge hebdomadaire – toutes activités")

weekly = (
    df.groupby(["week_label"])
    .agg(total_kcal=("calories", "sum"))
    .reset_index()
    .sort_values("week_label")
)

chart_week = (
    alt.Chart(weekly)
    .mark_line(point=True)
    .encode(
        x=alt.X("week_label:N", title="Semaine"),
        y=alt.Y("total_kcal:Q", title="Calories totales (kcal)"),
        tooltip=["week_label", "total_kcal"],
    )
    .properties(height=300)
)

st.altair_chart(chart_week, use_container_width=True)

st.subheader("📊 Charge hebdomadaire par sport")

weekly_sport = (
    df.groupby(["week_label", "sport"])
    .agg(total_kcal=("calories", "sum"))
    .reset_index()
    .sort_values(["week_label", "sport"])
)

chart_sport = (
    alt.Chart(weekly_sport)
    .mark_line(point=True)
    .encode(
        x=alt.X("week_label:N", title="Semaine"),
        y=alt.Y("total_kcal:Q", title="Calories (kcal)"),
        color=alt.Color("sport:N", title="Sport"),
        tooltip=["week_label", "sport", "total_kcal"],
    )
    .properties(height=350)
)

st.altair_chart(chart_sport, use_container_width=True)

# ----- Répartition par sport -----

st.subheader("🥧 Répartition des calories par sport (période affichée)")

by_sport = (
    df.groupby("sport")
    .agg(total_kcal=("calories", "sum"))
    .reset_index()
    .sort_values("total_kcal", ascending=False)
)

pie = (
    alt.Chart(by_sport)
    .mark_arc()
    .encode(
        theta="total_kcal:Q",
        color="sport:N",
        tooltip=["sport", "total_kcal"],
    )
)

st.altair_chart(pie, use_container_width=True)

st.dataframe(
    by_sport.rename(columns={"sport": "Sport", "total_kcal": "Calories (kcal)"}),
    use_container_width=True,
)

# ----- Dernières activités -----

st.markdown("---")
st.subheader("📋 Dernières activités")

df_display = df.sort_values("start_date", ascending=False).copy()
df_display["date"] = df_display["start_date"].dt.strftime("%Y-%m-%d %H:%M")
df_display["temps (min)"] = df_display["moving_time_min"].round(1)
df_display["distance (km)"] = df_display["distance_km"].round(2)
df_display["calories (kcal)"] = df_display["calories"].round(0)

st.dataframe(
    df_display[["date", "sport", "name", "distance (km)", "temps (min)", "calories (kcal)"]],
    use_container_width=True,
    height=300,
)

# ----- Bloc rapport à copier-coller -----

st.markdown("---")
st.subheader("📝 Rapport 'Miguel-ready' à copier-coller")

report_text = build_text_report(df, summary_7, summary_30, acwr)
st.text_area(
    "Rapport brut",
    value=report_text,
    height=400,
)
st.caption(
    "Quand tu veux qu'on analyse, tu copies ce bloc dans la discussion avec Miguel."
)
