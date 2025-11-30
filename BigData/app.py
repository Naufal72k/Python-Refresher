import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import os

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Smart Waste Mataram Command Center",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# BAGIAN 1: DATA GENERATION LAYER (SIMULASI SENSOR CERDAS)
# ==========================================

# Koordinat Referensi Mataram & Profil Wilayah
# multiplier: Faktor pengali kecepatan sampah penuh (Pasar > Perumahan)
LOCATIONS = [
    {"id": "TPS-01", "nama": "Ampenan Selatan", "lat": -8.5724, "lon": 116.0710, "type": "Residential", "multiplier": 0.8},
    {"id": "TPS-02", "nama": "Taman Sari", "lat": -8.5680, "lon": 116.0850, "type": "Residential", "multiplier": 0.9},
    {"id": "TPS-03", "nama": "Pejeruk", "lat": -8.5750, "lon": 116.0800, "type": "Residential", "multiplier": 1.0},
    {"id": "TPS-04", "nama": "Kebun Sari", "lat": -8.5800, "lon": 116.0880, "type": "Business", "multiplier": 1.2},
    {"id": "TPS-05", "nama": "Pagutan", "lat": -8.6000, "lon": 116.1050, "type": "Residential", "multiplier": 1.1},
    {"id": "TPS-06", "nama": "Pagesangan", "lat": -8.6100, "lon": 116.1100, "type": "Business", "multiplier": 1.3},
    {"id": "TPS-07", "nama": "Mataram Timur", "lat": -8.5830, "lon": 116.1200, "type": "Residential", "multiplier": 1.0},
    {"id": "TPS-08", "nama": "Selagalas", "lat": -8.5750, "lon": 116.1400, "type": "Residential", "multiplier": 0.9},
    {"id": "TPS-09", "nama": "Cakranegara", "lat": -8.5900, "lon": 116.1300, "type": "Market", "multiplier": 1.8}, # Pusat Bisnis/Pasar (Cepat Penuh)
    {"id": "TPS-10", "nama": "Bertais", "lat": -8.5950, "lon": 116.1500, "type": "Market", "multiplier": 1.9}, # Terminal/Pasar
    {"id": "TPS-11", "nama": "Sayang-Sayang", "lat": -8.5600, "lon": 116.1300, "type": "Residential", "multiplier": 0.8},
    {"id": "TPS-12", "nama": "Rembiga", "lat": -8.5550, "lon": 116.1100, "type": "Residential", "multiplier": 0.9},
    {"id": "TPS-13", "nama": "Dasan Agung", "lat": -8.5850, "lon": 116.1000, "type": "Density", "multiplier": 1.4},
    {"id": "TPS-14", "nama": "Gomong", "lat": -8.5900, "lon": 116.0950, "type": "Education", "multiplier": 1.2},
    {"id": "TPS-15", "nama": "Kekalik", "lat": -8.5980, "lon": 116.0900, "type": "Education", "multiplier": 1.3},
]

# Lokasi TPA Kebon Kongok
TPA_LOCATION = {"nama": "TPA Kebon Kongok", "lat": -8.6400, "lon": 116.1300}

@st.cache_data
def generate_data():
    """
    Simulasi Data Sensor IoT:
    - Menghasilkan data history 7 hari ke belakang.
    - Pola data mengikuti jam sibuk (Peak Hour) dan profil wilayah.
    """
    print("Generating Smart Data...")
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    current_time = start_date
    while current_time <= end_date:
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5
        
        # Faktor Waktu: Jam 17-20 sampah naik drastis, Jam 0-5 stagnan
        time_factor = 0
        if 6 <= hour < 12: time_factor = 5
        elif 12 <= hour < 17: time_factor = 10
        elif 17 <= hour <= 21: time_factor = 25 # Peak hour
        else: time_factor = 0 # Malam hari

        # Faktor Weekend: Sampah naik 20% saat weekend
        weekend_factor = 1.2 if is_weekend else 1.0

        for loc in LOCATIONS:
            # Base logic
            noise = np.random.randint(-5, 10)
            
            # Simulasi akumulasi sampah harian (reset setiap jam 5 pagi setelah diangkut)
            if hour == 5:
                base_fill = np.random.randint(0, 10) # Bersih pagi hari
            else:
                # Kepenuhan bertambah seiring waktu + faktor lokasi
                base_fill = (hour * 2) * loc['multiplier'] * weekend_factor + time_factor + noise
            
            # Logic khusus untuk membuat data 'Latest' bervariasi (Merah/Kuning/Hijau)
            if current_time == end_date:
                # Kita paksa persebaran data agar dashboard terlihat menarik
                if loc['multiplier'] > 1.5: base_fill = np.random.randint(80, 100) # Pasti Merah
                elif loc['multiplier'] > 1.1: base_fill = np.random.randint(50, 79) # Kemungkinan Kuning
                else: base_fill = np.random.randint(10, 49) # Hijau
            
            level_kepenuhan = max(0, min(int(base_fill), 100))
            
            # Korelasi Berat (kg) dengan Level (%)
            # Rumus: (Level * Kapasitas Tong 200kg / 100) + variasi sampah basah/kering
            berat_kg = (level_kepenuhan * 2.5) * np.random.uniform(0.9, 1.2)
            
            # Status Tutup & Bau
            status_tutup = "Buka" if level_kepenuhan > 85 else np.random.choice(["Tutup", "Buka"], p=[0.9, 0.1])
            status_bau = "Menyengat" if (status_tutup == "Buka" and level_kepenuhan > 70) else "Normal"

            data.append({
                "timestamp": current_time,
                "id_tps": loc['id'],
                "nama_lokasi": loc['nama'],
                "latitude": loc['lat'],
                "longitude": loc['lon'],
                "type": loc['type'],
                "level_kepenuhan": level_kepenuhan,
                "berat_kg": round(berat_kg, 2),
                "status_tutup": status_tutup,
                "status_bau": status_bau
            })
        
        current_time += timedelta(hours=1)

    df = pd.DataFrame(data)
    return df

# Load Data
df_raw = generate_data()

# ==========================================
# BAGIAN 2: PROCESSING & INTELLIGENCE LAYER
# ==========================================

# Ambil snapshot data terakhir untuk Real-time View
latest_timestamp = df_raw['timestamp'].max()
df_latest = df_raw[df_raw['timestamp'] == latest_timestamp].copy()

# A. ALGORITMA WARNA (Traffic Light Logic)
def get_status_color(level):
    if level > 75:
        return [255, 0, 0, 200]    # Merah (Bahaya)
    elif level > 50:
        return [255, 215, 0, 200]  # Kuning (Waspada)
    else:
        return [0, 255, 0, 200]    # Hijau (Aman)

df_latest['color'] = df_latest['level_kepenuhan'].apply(get_status_color)

# B. ALGORITMA CLUSTERING (K-Means)
# Mengelompokkan wilayah berdasarkan pola berat dan kepenuhan
if len(df_latest) > 3:
    X = df_latest[['level_kepenuhan', 'berat_kg']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_latest['cluster'] = kmeans.fit_predict(X)
    
    # Mapping cluster ke label Prioritas (High, Medium, Low) berdasarkan rata-rata level
    cluster_avg = df_latest.groupby('cluster')['level_kepenuhan'].mean().sort_values(ascending=False)
    priority_map = {
        cluster_avg.index[0]: '🔥 High Priority',
        cluster_avg.index[1]: '⚠️ Medium Priority',
        cluster_avg.index[2]: '✅ Low Priority'
    }
    df_latest['priority_status'] = df_latest['cluster'].map(priority_map)
else:
    df_latest['priority_status'] = 'Unknown'

# ==========================================
# BAGIAN 3: PRESENTATION LAYER (UI)
# ==========================================

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3299/3299901.png", width=50)
st.sidebar.title("Smart Waste Mataram")
st.sidebar.markdown("Sistem Manajemen Sampah Terintegrasi IoT & AI")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi Utama", 
    ["📡 Dashboard Monitoring", "🚚 Rute Pengangkutan", "📈 Analisis Data"]
)

st.sidebar.markdown("---")
# Filter Global
st.sidebar.subheader("⚙️ Filter Wilayah")
filter_kecamatan = st.sidebar.selectbox("Pilih Zona", ["Semua Zona", "Zona Bisnis", "Zona Perumahan", "Zona Pasar"])
st.sidebar.info(f"Last Sync: {latest_timestamp.strftime('%d %b %Y, %H:%M')} WITA")

# Filter Logic pada Dataframe
if filter_kecamatan == "Zona Bisnis":
    df_display = df_latest[df_latest['type'] == 'Business']
elif filter_kecamatan == "Zona Perumahan":
    df_display = df_latest[df_latest['type'] == 'Residential']
elif filter_kecamatan == "Zona Pasar":
    df_display = df_latest[df_latest['type'] == 'Market']
else:
    df_display = df_latest

# --- HALAMAN 1: DASHBOARD MONITORING ---
if "Dashboard" in menu:
    st.title("📡 Real-time Monitoring Dashboard")
    st.markdown("Pantauan langsung kondisi TPS di seluruh Kota Mataram.")

    # 1. Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_berat = df_display['berat_kg'].sum()
    avg_fullness = df_display['level_kepenuhan'].mean()
    critical_count = len(df_display[df_display['level_kepenuhan'] > 75])
    smell_count = len(df_display[df_display['status_bau'] == 'Menyengat'])

    col1.metric("Total Sampah (Saat Ini)", f"{total_berat/1000:.2f} Ton", "+0.5 Ton")
    col2.metric("Rata-rata Kapasitas", f"{avg_fullness:.1f}%", f"{avg_fullness-50:.1f}%")
    col3.metric("TPS Status Merah", f"{critical_count} Lokasi", "Butuh Pickup", delta_color="inverse")
    col4.metric("Laporan Bau", f"{smell_count} Lokasi", "Cek Filter", delta_color="inverse")

    # 2. Main Map
    st.subheader("Peta Sebaran & Status Kepenuhan")
    
    # Legend
    st.caption("Kriteria: 🔴 Penuh (>75%) | 🟡 Waspada (50-75%) | 🟢 Aman (<50%)")

    layer_scatter = pdk.Layer(
        "ScatterplotLayer",
        df_display,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="level_kepenuhan * 4", # Radius dinamis berdasarkan isi
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_min_pixels=5,
        radius_max_pixels=20,
    )

    view_state = pdk.ViewState(
        latitude=-8.5830,
        longitude=116.1100,
        zoom=12.5,
        pitch=45
    )

    tooltip = {
        "html": "<b>{nama_lokasi}</b><br/>"
                "Status: <b>{priority_status}</b><br/>"
                "Isi: {level_kepenuhan}%<br/>"
                "Berat: {berat_kg} Kg<br/>"
                "Bau: {status_bau}",
        "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "12px", "padding": "10px"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=[layer_scatter],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/light-v10'
    ))

    # 3. Alert Table (Critical Only)
    col_table, col_detail = st.columns([2, 1])
    
    with col_table:
        st.subheader("⚠️ Peringatan Dini (Action Needed)")
        critical_df = df_display[df_display['level_kepenuhan'] > 50].sort_values('level_kepenuhan', ascending=False)
        
        if not critical_df.empty:
            # Formatting dataframe agar cantik
            st.dataframe(
                critical_df[['nama_lokasi', 'level_kepenuhan', 'berat_kg', 'priority_status', 'status_bau']],
                use_container_width=True,
                column_config={
                    "level_kepenuhan": st.column_config.ProgressColumn("Level (%)", format="%d%%", min_value=0, max_value=100),
                    "priority_status": "Prioritas",
                    "status_bau": "Kondisi Udara"
                }
            )
        else:
            st.success("Semua TPS dalam kondisi aman (Hijau).")

    with col_detail:
        st.info("💡 **AI Insight:**\nTPS Cakranegara menunjukkan anomali kenaikan 20% lebih cepat dari biasanya hari ini karena aktivitas pasar.")

# --- HALAMAN 2: RUTE PENGANGKUTAN ---
elif "Rute" in menu:
    st.title("🚚 Smart Routing System")
    st.markdown("Optimasi rute pengangkutan berdasarkan prioritas kepenuhan sampah.")

    # Algoritma Filter Greedy: Ambil hanya yang > 75%
    pickup_list = df_latest[df_latest['level_kepenuhan'] > 75].sort_values(by='level_kepenuhan', ascending=False)
    
    if pickup_list.empty:
        st.success("🎉 Tidak ada TPS Kritis. Armada bisa standby.")
    else:
        st.warning(f"Terdeteksi **{len(pickup_list)} Titik Kritis** yang harus segera diangkut!")

        # Layout Column
        c1, c2 = st.columns([1, 2])

        with c1:
            st.write("### 📋 Urutan Penjemputan")
            st.caption("Diurutkan berdasarkan tingkat urgensi:")
            
            route_coords = []
            
            for i, (index, row) in enumerate(pickup_list.iterrows()):
                st.markdown(f"**{i+1}. {row['nama_lokasi']}**")
                st.progress(row['level_kepenuhan']/100)
                st.caption(f"Isi: {row['level_kepenuhan']}% | Berat: {row['berat_kg']} Kg")
                route_coords.append([row['longitude'], row['latitude']])
            
            st.markdown("⬇️")
            st.markdown(f"🏁 **{TPA_LOCATION['nama']}** (Final Dump)")
            # Tambah TPA ke koordinat
            route_coords.append([TPA_LOCATION['lon'], TPA_LOCATION['lat']])

        with c2:
            st.write("### 🗺️ Visualisasi Rute Efektif")
            
            # Membuat Garis Rute (LineLayer)
            lines = []
            for i in range(len(route_coords) - 1):
                lines.append({
                    "start": route_coords[i],
                    "end": route_coords[i+1],
                    "name": f"Segmen {i+1}"
                })

            layer_lines = pdk.Layer(
                "LineLayer",
                lines,
                get_source_position="start",
                get_target_position="end",
                get_color=[50, 50, 200], # Biru
                get_width=5,
                pickable=True
            )

            # Layer Titik Jemput
            layer_points = pdk.Layer(
                "ScatterplotLayer",
                pickup_list,
                get_position=["longitude", "latitude"],
                get_color=[255, 0, 0, 200], # Merah
                get_radius=300,
                pickable=True
            )

            # Layer TPA
            df_tpa = pd.DataFrame([TPA_LOCATION])
            layer_tpa = pdk.Layer(
                "ScatterplotLayer",
                df_tpa,
                get_position=["lon", "lat"],
                get_color=[0, 0, 0, 200], # Hitam
                get_radius=500,
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(
                layers=[layer_lines, layer_points, layer_tpa],
                initial_view_state=pdk.ViewState(lat=-8.6000, lon=116.1100, zoom=11.5),
                tooltip={"text": "Jalur Armada Kebersihan"}
            ))

# --- HALAMAN 3: ANALISIS DATA ---
elif "Analisis" in menu:
    st.title("📈 Analisis & Prediksi Cerdas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Tren Historis", "🏘️ Perbandingan Wilayah", "🔮 AI Prediction"])

    with tab1:
        st.subheader("Volume Sampah 7 Hari Terakhir")
        # Grouping data per jam untuk grafik line
        hourly_data = df_raw.groupby('timestamp')[['berat_kg']].sum().reset_index()
        st.area_chart(hourly_data.set_index('timestamp'), color="#2e7d32")
        st.caption("Pola grafik menunjukkan kenaikan signifikan pada sore hari (Jam Pulang Kerja).")

    with tab2:
        st.subheader("Produktivitas Sampah per Wilayah")
        # Bar chart total sampah per lokasi
        loc_stats = df_raw.groupby('nama_lokasi')['berat_kg'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(loc_stats, color="#ffaa00")
        st.caption("10 Wilayah penyumbang sampah terbesar minggu ini.")

    with tab3:
        st.subheader("Prediksi Total Sampah Harian (Regression)")
        
        # Agregasi data harian
        daily_df = df_raw.groupby(df_raw['timestamp'].dt.date)['berat_kg'].sum().reset_index()
        daily_df['hari_ke'] = range(len(daily_df))
        
        # Simple Linear Regression (y = mx + c)
        x = daily_df['hari_ke']
        y = daily_df['berat_kg']
        coef = np.polyfit(x, y, 1) # Derajat 1 (Linear)
        poly1d_fn = np.poly1d(coef)
        
        # Prediksi Hari Esok
        next_day_index = len(daily_df)
        prediction_val = poly1d_fn(next_day_index)
        
        c_pred1, c_pred2 = st.columns(2)
        c_pred1.metric("Rata-rata Harian", f"{y.mean():.2f} Kg")
        c_pred2.metric("Prediksi Besok", f"{prediction_val:.2f} Kg", f"{(prediction_val - y.iloc[-1]):.2f} Kg", delta_color="inverse")
        
        # Plotting
        daily_df['Prediksi Trend'] = poly1d_fn(x)
        chart_data = daily_df.set_index('timestamp')[['berat_kg', 'Prediksi Trend']]
        st.line_chart(chart_data)
        st.info("Garis oranye menunjukkan tren prediksi linear. Jika tren naik, armada tambahan mungkin diperlukan besok.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: grey;'>
        <small>Smart Waste Management System v2.0 &copy; 2025 Kota Mataram<br>
        Powered by Streamlit, PyDeck, & Scikit-Learn</small>
    </div>
    """, unsafe_allow_html=True)