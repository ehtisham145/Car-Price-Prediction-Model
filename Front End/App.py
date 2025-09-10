# streamlit_car_price_app.py
# Streamlit frontend for Car Price prediction
# Place your trained pipeline (joblib) in the same folder as 'car_model.pkl' or upload via sidebar.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# CSS for light/dark + layout
# -------------------------
BASE_CSS = """
<style>
:root{ --bg:#f7fafc; --card:#ffffff; --text:#0f172a; --muted:#6b7280; --accent:#0ea5e9; }
body{ background:var(--bg); color:var(--text); }
.header{ padding:1rem 2rem; background:linear-gradient(90deg,#0ea5e9,#7c3aed); color:white; border-radius:12px;}
.navbar{ display:flex; gap:1rem; align-items:center; }
.nav-item{ padding:0.45rem 0.9rem; background:rgba(255,255,255,0.08); border-radius:8px; font-weight:600;}
.card{ background:var(--card); padding:1rem; border-radius:12px; box-shadow:0 6px 24px rgba(2,6,23,0.06);}
.footer{ padding:1rem; text-align:center; color:var(--muted); }
.small{ font-size:0.9rem; color:var(--muted); }
.btn{ background:var(--accent); color:white; padding:0.55rem 1rem; border-radius:10px; font-weight:700; }
</style>
"""

DARK_CSS = """
<style>
:root{ 
  --bg:#000000;           /* Pure black background */
  --card:#111111;         /* Dark gray cards */
  --text:#ffffff;         /* White text */
  --muted:#aaaaaa;        /* Muted gray for less important text */
  --accent:#0ea5e9;       /* Blue accent */
}

body, .stApp { 
  background-color: var(--bg) !important; 
  color: var(--text) !important;
}

/* 🔥 Header black in dark mode */
.header{ 
  padding:1rem 2rem; 
  background:#000000; 
  color:white; 
  border-radius:12px;
}

.navbar{ display:flex; gap:1rem; align-items:center; }
.nav-item{ 
  padding:0.45rem 0.9rem; 
  background:rgba(255,255,255,0.1); 
  border-radius:8px; 
  font-weight:600; 
  color:white;
}

.card{ 
  background:var(--card); 
  padding:1rem; 
  border-radius:12px; 
  box-shadow:0 6px 24px rgba(255,255,255,0.1);
  color: var(--text);
}

.footer{ 
  padding:1rem; 
  text-align:center; 
  color:var(--muted); 
}

.small{ font-size:0.9rem; color:var(--muted); }

.btn{ 
  background:var(--accent); 
  color:white; 
  padding:0.55rem 1rem; 
  border-radius:10px; 
  font-weight:700; 
}

/* 🔥 Brighten input fields */
.stTextInput > div > div > input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
  background-color: #1a1a1a !important;
  color: #ffffff !important;
  border: 1px solid #555555 !important;
  font-weight: 600 !important;
}

.stSelectbox [data-baseweb="select"] span {
  color: #ffffff !important;
}

h2, h3 {
  background: linear-gradient(90deg, #0ea5e9, #7c3aed);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
/* 🎨 Table ko visible banane ke liye */
thead tr th {
    background-color: #111111 !important; 
    color: #ffffff !important; 
    font-weight: 700 !important;
}

tbody tr td {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: 1px solid #333333 !important;
}

/* 🎨 Predict Button Style */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #7c3aed) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.5) !important;
    transition: all 0.3s ease-in-out !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #0ea5e9) !important;
    transform: scale(1.05) !important;
}/* 🎨 Input Summary Table Fix */
.stTable tr th, .stTable tr td {
    background-color: #1a1a1a !important;  /* thoda dark grey */
    color: #f5f5f5 !important;             /* white-ish text */
    font-weight: 700 !important;           /* bold for visibility */
    font-size: 16px !important;            /* thoda bada text */
    border: 1px solid #444 !important;     /* thoda border for separation */
    padding: 8px 12px !important;          /* extra spacing for clarity */
}

</style>
"""
# Session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

def toggle_dark():
    st.session_state['dark_mode'] = not st.session_state['dark_mode']

st.markdown(DARK_CSS if st.session_state['dark_mode'] else BASE_CSS, unsafe_allow_html=True)

# -------------------------
# Top navbar + header
# -------------------------
with st.container():
    st.markdown("""
    <div class='header'>
      <div class='navbar'>
        <div style='font-size:1.4rem; font-weight:800;'>Car Price Predictor</div>
        <div style='flex:1'></div>
        <div class='nav-item'>Home</div>
        <div class='nav-item'>About</div>
        <div class='nav-item'>Model Info</div>
      </div>
      <div style='margin-top:0.5rem; color:rgba(255,255,255,0.95);'>A sleek Streamlit app to predict car price using a trained pipeline.</div>
    </div>
    """, unsafe_allow_html=True)

# Small intro about you (editable)
st.markdown("""
<div class='card'>
  <h3>About the developer</h3>
  <p class='small'>Hi! I'm Ehtisham, currently learning and growing in the field of AI. My goal is to become a Gen AI Engineer and build impactful AI solutions. This app is one of my learning projects where I connect Machine Learning models with Streamlit frontends.</p>
</div>
""", unsafe_allow_html=True)

st.write("\n")

# -------------------------
# Sidebar inputs
# -------------------------
sb = st.sidebar
sb.title("Input Controls")
sb.write("Provide car details to predict price")

sb.checkbox("Dark mode", value=st.session_state['dark_mode'], key='dark_mode_cb', on_change=toggle_dark)

sb.markdown("---")

# Presets
sb.subheader("Quick presets")
preset = sb.selectbox("Choose preset", ["Custom","Economy small","Family sedan","Luxury automatic"]) 

if preset == "Economy small":
    preset_vals = {'year':2012, 'km_driven':50000, 'fuel':'Petrol', 'transmission':'Manual', 'max_power':68.0}
elif preset == "Family sedan":
    preset_vals = {'year':2016, 'km_driven':30000, 'fuel':'Diesel', 'transmission':'Manual', 'max_power':110.0}
elif preset == "Luxury automatic":
    preset_vals = {'year':2020, 'km_driven':15000, 'fuel':'Petrol', 'transmission':'Automatic', 'max_power':250.0}
else:
    preset_vals = {}

# Input widgets
year = sb.number_input("Year", min_value=1990, max_value=2025, value=preset_vals.get('year',2015))
km_driven = sb.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=preset_vals.get('km_driven',50000))

fuel = sb.selectbox("Fuel", options=['Diesel','Petrol','LPG','CNG'], index=['Diesel','Petrol','LPG','CNG'].index(preset_vals.get('fuel','Petrol')))
transmission = sb.selectbox("Transmission", options=['Manual','Automatic'], index=['Manual','Automatic'].index(preset_vals.get('transmission','Manual')))
max_power = sb.number_input("Max Power (bhp)", min_value=10.0, max_value=1000.0, value=preset_vals.get('max_power',80.0), format="%.1f")

sb.markdown("---")

# Model upload
sb.subheader("Model")
model_file = sb.file_uploader("Upload trained pipeline (.pkl/.joblib)", type=['pkl','joblib'])

# -------------------------
# Main area: show inputs and prediction
# -------------------------
st.markdown("""
<div class='card'>
  <h2>About this app</h2>
  <p class='small'>This app predicts the expected selling price of a car using features year, km_driven, fuel, transmission and max_power. Upload your trained scikit-learn pipeline (including preprocessing) as 'car_model.pkl' or use the uploader in the sidebar.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### Input summary")
    input_df = pd.DataFrame([{
        'year': int(year),
        'km_driven': int(km_driven),
        'fuel': fuel,
        'transmission': transmission,
        'max_power': float(max_power)
    }])
    st.table(input_df.T)
    predict_btn = st.button("Predict Price")

with col2:
    st.markdown("### Model Info")
    st.write("This model will predict the price of car based on the features present in sidebar")

# Load model
model = None
if model_file is not None:
    tmp_path = "uploaded_car_model.pkl"
    with open(tmp_path, "wb") as f:
        f.write(model_file.getbuffer())
    try:
        model = joblib.load(tmp_path)
        st.success("Model uploaded and loaded.")
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")
else:
    try:
        model = joblib.load("car_model.pkl")
        st.info("Loaded 'car_model.pkl' from app folder.")
    except Exception:
        st.warning("No model found. Upload a trained pipeline in the sidebar.")

# Prediction
if predict_btn:
    if model is None:
        st.error("No model loaded. Upload or place 'car_model.pkl' in app folder.")
    else:
        try:
            # enforce dtypes
            input_df = input_df.astype({'year':'int64','km_driven':'int64','fuel':'object','transmission':'object','max_power':'float64'})

            pred = model.predict(input_df)
            st.success(f"Predicted price: {pred[0]:.2f}")

            # show transformed features if accessible
            if hasattr(model, 'named_steps'):
                try:
                    # try to find preprocess step
                    if 'preprocessor' in model.named_steps:
                        transformed = model.named_steps['preprocessor'].transform(input_df)
                        st.write('Transformed features preview:')
                        st.write(np.asarray(transformed))
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.markdown("\n")
st.markdown("""
<div class='footer'>
  Built with ❤️ using Streamlit · Car Price Predictor · Make sure your pipeline contains the same feature names and preprocessing.
</div>
""", unsafe_allow_html=True)
