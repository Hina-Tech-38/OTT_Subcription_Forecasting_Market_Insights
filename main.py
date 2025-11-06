import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# ---------- Page setup ----------
st.set_page_config(
    page_title="Forecasting Netflix Subscription",
    page_icon="🎬",
    layout="wide"
)

# ---------- Netflix theme (custom CSS) ----------
st.markdown("""
<style>
/* Background + fonts */
html, body, [class*="css"]  {
  color: #e5e5e5 !important;
  background-color: #0b0b0b !important;
  font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* Main container width & spacing */
.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
  max-width: 1200px;
}

/* Title styling */
h1, h2, h3 {
  color: #ffffff !important;
  letter-spacing: 0.2px;
}
h1 span.brand {
  color: #e50914;
  font-weight: 800;
}

/* Card styles */
.card {
  background: #141414;
  border: 1px solid #262626;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 0 0 1px rgba(229, 9, 20, 0.12), 0 10px 30px rgba(0,0,0,0.35);
}
.card subtle {
  color: #a3a3a3;
}
/All title
.red-title {
  color: #E50914 !important;
  font-weight: 700 !important;
  font-size: 22px !important;
  margin-top: 15px;
  margin-bottom: 10px;
}

/* Metric cards */
.metric {
  background: #141414;
  border: 1px solid #262626;
  border-radius: 14px;
  padding: 16px 18px;
  text-align: center;
}
.metric .label { color:#a3a3a3; font-size: 13px; }
.metric .value { color:#fff; font-weight: 700; font-size: 22px; }

/* Buttons & widgets */
.stButton>button, .stDownloadButton>button {
  background: #e50914 !important;
  color: white !important;
  border-radius: 999px !important;
  border: none !important;
  padding: 0.6rem 1.1rem !important;
  font-weight: 700 !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
  background: #f6121d !important;
}

/* File uploader */
.css-1vq4p4l, .st-emotion-cache-1dm2o9w { /* container tweaks may vary by version */
  background: #141414 !important;
  border: 1px dashed #333 !important;
  border-radius: 14px !important;
}

/* Tables */
.dataframe {
  filter: contrast(105%);
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(
    "<h1>🎬 <span class='brand'>NETFLIX</span> Subscription Growth Forecast</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='card'><span style='color:white; font-weight:700;'>Use historical subscriber data to forecast <span style='color:#E50914;'>future growth</span> with time series modeling.</span></div>",
    unsafe_allow_html=True
)

#st.markdown(
 #   "<div class='card'>Use historical subscriber data to forecast future growth with time series modeling.</div>",
   # unsafe_allow_html=True
#)

st.write("")

# ---------- Sidebar (controls) ----------
with st.sidebar:
    st.markdown("<div style='color:#000000; font-size:25px; font-weight:800;'>⚙️ Controls </div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Mock_Netflix_Subscription_Growth_2014_2024.csv", type=["csv"])
    horizon_quarters = st.slider("Forecast horizon (quarters)", 4, 24, 8, step=4)
    show_components = st.checkbox("Show trend/seasonality components", value=True)
    st.markdown("---")
    st.caption("Tip: Dates can be in `MM/DD/YYYY` format or quarter-end dates; we’ll parse them automatically.")

#ABOUT SECTION
    st.markdown("---")
    st.markdown("<div style='color:#000000; font-size:25px; font-weight:800;'>🎬 About This App </div>", unsafe_allow_html=True)
    st.write("""
    This interactive dashboard demonstrates how **time series forecasting**
    can be applied to predict **Netflix subscriber growth** over time.  
    The model uses **ARIMA**, a forecasting library developed by George Box and Gwilym Jenkins,
    to identify long-term trends, growth patterns, to the data which shows less to no seasonality.
    """)
    st.markdown("---")

#DATA SOURCE SECTION
    st.markdown("<div style='color:#000000; font-size:25px; font-weight:800;'>📊 Data Source </div>", unsafe_allow_html=True)
    st.write("""
    The dataset used in this demo is a **fictional Netflix subscriptions growth dataset**
    covering quarterly data from **2014 to 2024**.  
    It includes the number of active subscribers recorded at the start of each quarter.
    """)

    st.markdown("---")

# ---------- Data load ----------
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    assert "Time period" in df.columns and "Subscribers" in df.columns, \
        "CSV must include columns: 'Time period' and 'Subscribers'."

    # Parse dates: handle "Qx YYYY" or "MM/DD/YYYY"
    def parse_date(s: str):
        s = str(s).strip()
        if s.upper().startswith("Q"):
            # Convert "Q1 2019" -> quarter end date
            q, y = s.upper().split()
            qn = int(q.replace("Q", ""))
            year = int(y)
            # quarter end months: 3,6,9,12
            month = [3,6,9,12][qn-1]
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        else:
            # Try MDY then ISO as fallback
            try:
                return pd.to_datetime(s, format="%m/%d/%Y")
            except:
                return pd.to_datetime(s, errors="coerce")

    df["ds"] = df["Time period"].apply(parse_date)
    if df["ds"].isna().any():
        bad = df[df["ds"].isna()]["Time period"].head(3).tolist()
        raise ValueError(f"Unparseable dates found. Examples: {bad}")

    # Subscribers to numeric
    df["y"] = pd.to_numeric(df["Subscribers"], errors="coerce")
    if df["y"].isna().any():
        raise ValueError("Some 'Subscribers' values are not numeric.")

    # Sort and keep necessary cols
    df = df[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    # Fallback message + gentle sample generator
    st.info("No file uploaded. Using an example quarterly dataset (2014–2024).")
    rng = pd.date_range("2014-03-31", "2024-12-31", freq="Q")
    np.random.seed(42)
    subs = 20_000_000
    series = []
    for _ in rng:
        subs = int(subs * (1 + np.random.uniform(0.03, 0.08)))
        series.append(subs)
    df = pd.DataFrame({"ds": rng, "y": series})

# ---------- KPIs ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='metric'><div class='label'>First date</div>"
                f"<div class='value'>{df['ds'].min().date()}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric'><div class='label'>Latest date</div>"
                f"<div class='value'>{df['ds'].max().date()}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric'><div class='label'>Latest subscribers</div>"
                f"<div class='value'>{int(df['y'].iloc[-1]):,}</div></div>", unsafe_allow_html=True)

st.write("")

# ---------- Historical chart ----------
st.markdown("<div style='color:#E50914; font-size:22px; font-weight:700;'>📈 Historical Trend</div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["ds"], df["y"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Subscribers")
    ax.set_title("Historical Netflix Subscription Growth")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Forecast ----------
st.markdown("<div style='color:#E50914; font-size:22px; font-weight:700;'>🔮 Forecast </div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon_quarters, freq="Q")
    forecast = model.predict(future)

    # Table
    tail_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    nice = forecast[tail_cols].tail(horizon_quarters).copy()
    nice.columns = ["Date", "Forecast", "Lower", "Upper"]
    st.dataframe(nice.style.format({
        "Forecast": "{:,.0f}",
        "Lower": "{:,.0f}",
        "Upper": "{:,.0f}"
    }))

    # Plot forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    if show_components:
        st.markdown("#### Trend & Seasonality")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    "<br><div style='text-align:center; color:#a3a3a3;'>Built with ❤️ By Heena Shaikh • Data Science & Business Analytics • "
    "Theme inspired by <span style='color:#e50914;'>Netflix</span></div>",
    unsafe_allow_html=True
)

