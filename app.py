import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0f172a;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
    }
    .css-1d391kg {
        background-color: #1e293b;
        border-radius: 15px;
        padding: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        transform: scale(1.05);
        transition: 0.3s;
    }
    .result-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #38bdf8;
        color: #e2e8f0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL AND DATA ---
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model/data: {e}")
    st.stop()

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#94a3b8;'>Predict the price of your dream laptop based on specifications 🧠</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("📋 About the App")
st.sidebar.info("""
This ML-powered app predicts **laptop prices** using specs like:
- Brand, CPU, RAM, GPU, Storage, etc.  
- Powered by a Machine Learning pipeline  
- Built with 🐍 Python & Streamlit
""")

st.sidebar.markdown("👨‍💻 **Developer:** Shahid Mulani")

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('🏢 Brand', df['Company'].unique())
    type_name = st.selectbox('💼 Type', df['TypeName'].unique())
    ram = st.selectbox('💾 RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('⚖️ Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('🖱️ Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('📺 IPS Display', ['No', 'Yes'])
    screen_size = st.slider('📏 Screen Size (inches)', 10.0, 18.0, 13.0)

with col2:
    resolution = st.selectbox('🔹 Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    cpu = st.selectbox('⚙️ CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('💽 HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('🔋 SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('🎮 GPU', df['Gpu brand'].unique())
    os = st.selectbox('🪟 Operating System', df['os'].unique())

# --- PREDICTION BUTTON ---
if st.button('🚀 Predict Price'):
    try:
        # convert inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # create dataframe with correct column names
        query = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'Ips': [ips],
            'ppi': [ppi],  # ✅ fixed lowercase key
            'Cpu brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu],
            'os': [os]
        })

        # predict price
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        # show result
        st.markdown("<div class='result-box'><h2>💰 Estimated Laptop Price</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color:#38bdf8;'>₹ {predicted_price:,}</h1></div>", unsafe_allow_html=True)
        st.balloons()

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
