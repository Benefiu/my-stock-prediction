import streamlit as st
import plotly.graph_objects as go
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib
from pymongo import MongoClient
import gridfs
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Tőzsdei Előrejelző", layout="wide")

# MongoDB URI kezelése
try:
    MONGO_URI = st.secrets["MONGO_URI"]
except:
    ()
client = MongoClient(MONGO_URI)
db = client["stock_prediction"]
fs = gridfs.GridFS(db)

# Backend Logika és MongoDB szinkronizáció

def sync_file_from_mongodb(filename, local_path):
    """Ellenőrzi és letölti a modellt a MongoDB-ből."""
    file_data = db.fs.files.find_one({"filename": filename})
    if not file_data:
        return False
    
    needs_download = not os.path.exists(local_path)
    if not needs_download:
        local_mtime = os.path.getmtime(local_path)
        db_mtime = file_data['uploadDate'].timestamp()
        if db_mtime > local_mtime:
            needs_download = True
            
    if needs_download:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(fs.get(file_data['_id']).read())
    return True

def predict_stock_logic(ticker, start_date_str, end_date_str, forecast_days, lookback_days=60):
    model_path = f"models/{ticker}_model.keras"
    scaler_x_path = f"models/{ticker}_scaler_X.pkl"
    scaler_y_path = f"models/{ticker}_scaler_y.pkl"
    
    # Szinkronizáció
    if not (sync_file_from_mongodb(f"{ticker}_model.keras", model_path) and 
            sync_file_from_mongodb(f"{ticker}_scaler_X.pkl", scaler_x_path) and 
            sync_file_from_mongodb(f"{ticker}_scaler_y.pkl", scaler_y_path)):
        return None, "Nincs modell az adatbázisban ehhez a tickerhez."

    model = load_model(model_path)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    fetch_start = (datetime.datetime.strptime(start_date_str, "%Y-%m-%d") - datetime.timedelta(days=40)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=fetch_start, end=end_date_str)
    
    if df.empty or len(df) < lookback_days + 20:
        return None, "Nincs elég adat az előrejelzéshez."
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df = df.dropna().loc[start_date_str:]
    
    last_window_data = df[['Close', 'SMA_20']].values[-lookback_days:]
    current_batch = scaler_X.transform(last_window_data).reshape(1, lookback_days, 2)
    
    future_prices = []
    current_closes_for_sma = df['Close'].values.flatten().tolist() 
    
    for _ in range(forecast_days):
        pred_scaled = model.predict(current_batch, verbose=0)
        pred_price = scaler_y.inverse_transform(pred_scaled)[0, 0]
        future_prices.append(float(pred_price))
        
        current_closes_for_sma.append(pred_price)
        new_sma = np.mean(current_closes_for_sma[-20:])
        new_step_scaled = scaler_X.transform(np.array([[pred_price, new_sma]])).reshape(1, 1, 2)
        current_batch = np.append(current_batch[:, 1:, :], new_step_scaled, axis=1)

    return {
        "hist_dates": df.index.strftime('%Y-%m-%d').tolist(),
        "hist_prices": df['Close'].values.flatten().tolist(),
        "fut_dates": pd.bdate_range(start=df.index[-1] + pd.offsets.BDay(1), periods=forecast_days).strftime('%Y-%m-%d').tolist(),
        "fut_prices": future_prices
    }, None

# Frontend

# CSS stílusok
st.markdown("""
<style>
 .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #ddd; }
    .trend-up { color: green; font-weight: bold; }
    .trend-down { color: red; font-weight: bold; }
    div[data-baseweb="select"] > div {cursor: pointer; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Beállítások")
ticker_names = {
    "AAPL": "APPLE (AAPL)",
    "TSLA": "TESLA (TSLA)",
    "MSFT": "MICROSOFT (MSFT)",
    "GOOGL": "GOOGLE (GOOGL)",
    "AMZN": "AMAZON (AMZN)"
}
ticker = st.sidebar.selectbox("Részvény", options=list(ticker_names.keys()), format_func=lambda x: ticker_names[x])

# ÚJ LOGIKA: Szimulált "Ma" bevezetése
start_date = st.sidebar.date_input("Múltbeli kezdőpont (Adatok lekérése innen)", datetime.date.today() - datetime.timedelta(days=90))
predict_start_date = st.sidebar.date_input("Előrejelzés indítása (Szimulált 'Ma')", datetime.date.today() - datetime.timedelta(days=5)) # Ezt adtuk hozzá
target_date = st.sidebar.date_input("Előrejelzés vége", datetime.date.today())

if st.sidebar.button("Futtatás"):
    # A valódi mai nap helyett a kiválasztott szimulált napot használjuk kiindulópontnak
    today_str = predict_start_date.strftime('%Y-%m-%d')
    target_str = target_date.strftime('%Y-%m-%d')
    
    # Kiszámoljuk a munkanapokat a két megadott dátum között
    forecast_days = int(np.busday_count(today_str, target_str)) + 1
    
    # A hibaüzenet is megváltozik, hogy a két bemenetet hasonlítsa össze
    if forecast_days <= 0:
        st.error("Az előrejelzés vége későbbi kell legyen, mint az indítás dátuma!")
    else:
        with st.spinner("Modell letöltése és számítás..."):
            res, err = predict_stock_logic(ticker, start_date.strftime("%Y-%m-%d"), today_str, min(forecast_days, 90))
            
            if err:
                st.error(err)
            else:
                # Kártyák és Grafikon
                col1, col2, col3 = st.columns(3)
                last_p, final_p = res["hist_prices"][-1], res["fut_prices"][-1]
                
                col1.metric("Ticker", ticker)
                col2.metric("Trend", "EMELKEDŐ" if final_p > last_p else "CSÖKKENŐ", f"{((final_p/last_p)-1)*100:.2f}%")
                col3.metric("Célár", f"{final_p:.2f} USD")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["hist_dates"], y=res["hist_prices"], name="Múltbeli ár"))
                fig.add_trace(go.Scatter(x=res["fut_dates"], y=res["fut_prices"], name="Előrejelzés", line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
