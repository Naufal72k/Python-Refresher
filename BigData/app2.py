"""
================================================================================
SMART WASTE MANAGEMENT SYSTEM - MATARAM CITY DASHBOARD
================================================================================
Author: AI Assistant
Date: 2023
Version: 1.0.0

DESKRIPSI SISTEM:
Sistem ini adalah simulasi end-to-end untuk manajemen sampah cerdas berbasis IoT.
Mencakup:
1. Data Generator (Simulasi Sensor IoT pada tong sampah)
2. Data Processing (Pembersihan dan Agregasi)
3. Intelligent Analytics (K-Means Clustering & Regresi Linear)
4. Interactive Dashboard (Streamlit UI)

FITUR UTAMA:
- Monitoring Real-time kepenuhan TPS.
- Peta Interaktif dengan indikator warna (Hijau/Kuning/Merah).
- Algoritma Optimasi Rute Pengangkutan.
- Prediksi timbulan sampah harian.
- Analisis Cluster wilayah prioritas.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import time
import random

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS (THEME: BRIGHT/TERANG)
# ==============================================================================

st.set_page_config(
    page_title="Smart Waste Mataram",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang bersih, terang, dan profesional
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E9ECEF;
    }
    
    /* Metrics Card styling */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #6C757D;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    div[data-testid="stMetricValue"] {
        color: #0D6EFD; /* Bootstrap Primary Blue */
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #198754; /* Green for eco-friendly vibe */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px;
        color: #495057;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #198754;
        color: #FFFFFF;
    }

    /* Table styling */
    .dataframe {
        font-size: 0.9rem !important;
    }

    /* Custom Alert Box */
    .alert-box {
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid transparent;
        border-radius: 4px;
    }
    
    .alert-danger {
        color: #842029;
        background-color: #f8d7da;
        border-color: #f5c2c7;
    }
    
    .alert-success {
        color: #0f5132;
        background-color: #d1e7dd;
        border-color: #badbcc;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODUL GENERATOR DATA (SIMULASI SENSOR IOT)
# ==============================================================================

class WasteDataGenerator:
    """
    Kelas ini bertanggung jawab untuk mensimulasikan data sensor IoT.
    Mengikuti spesifikasi teknis untuk menghasilkan dataset dummy yang realistis.
    """
    
    def __init__(self):
        # Koordinat TPS di sekitar Mataram (Dummy tapi realistis secara geografis)
        # Lat/Lon diambil di sekitar area Mataram, Lombok
        self.tps_locations = [
            {"id": "TPS-01", "nama": "Kec. Ampenan - Pasar", "lat": -8.575, "lon": 116.080, "base_load": 80},
            {"id": "TPS-02", "nama": "Kec. Ampenan - Pemukiman", "lat": -8.578, "lon": 116.085, "base_load": 40},
            {"id": "TPS-03", "nama": "Kec. Sekarbela - Pesisir", "lat": -8.595, "lon": 116.082, "base_load": 50},
            {"id": "TPS-04", "nama": "Kec. Sekarbela - Kampus", "lat": -8.592, "lon": 116.090, "base_load": 60},
            {"id": "TPS-05", "nama": "Kec. Mataram - Perkantoran", "lat": -8.583, "lon": 116.102, "base_load": 45},
            {"id": "TPS-06", "nama": "Kec. Mataram - Sekolah", "lat": -8.585, "lon": 116.105, "base_load": 55},
            {"id": "TPS-07", "nama": "Kec. Mataram - Taman Udayana", "lat": -8.570, "lon": 116.100, "base_load": 70},
            {"id": "TPS-08", "nama": "Kec. Selaparang - Rembiga", "lat": -8.565, "lon": 116.110, "base_load": 50},
            {"id": "TPS-09", "nama": "Kec. Selaparang - Bandara Lama", "lat": -8.562, "lon": 116.105, "base_load": 30},
            {"id": "TPS-10", "nama": "Kec. Cakranegara - Bisnis A", "lat": -8.590, "lon": 116.120, "base_load": 85},
            {"id": "TPS-11", "nama": "Kec. Cakranegara - Mall", "lat": -8.592, "lon": 116.125, "base_load": 90},
            {"id": "TPS-12", "nama": "Kec. Cakranegara - Pasar Induk", "lat": -8.588, "lon": 116.130, "base_load": 95},
            {"id": "TPS-13", "nama": "Kec. Cakranegara - Hotel", "lat": -8.595, "lon": 116.122, "base_load": 65},
            {"id": "TPS-14", "nama": "Kec. Sandubaya - Terminal", "lat": -8.600, "lon": 116.135, "base_load": 75},
            {"id": "TPS-15", "nama": "Kec. Sandubaya - Bertais", "lat": -8.598, "lon": 116.140, "base_load": 80},
            {"id": "TPS-16", "nama": "Kel. Pagutan - Perumahan", "lat": -8.605, "lon": 116.100, "base_load": 55},
            {"id": "TPS-17", "nama": "Kel. Pagutan - Pasar", "lat": -8.608, "lon": 116.102, "base_load": 70},
            {"id": "TPS-18", "nama": "Dasam Agung - Padat Penduduk", "lat": -8.580, "lon": 116.110, "base_load": 85},
            {"id": "TPS-19", "nama": "Gomong - Mahasiswa", "lat": -8.582, "lon": 116.095, "base_load": 60},
            {"id": "TPS-20", "nama": "Sayang-Sayang - Kuliner", "lat": -8.560, "lon": 116.120, "base_load": 75},
        ]

    def generate_data(self, days=7):
        """
        Menghasilkan data sensor per jam selama n hari ke belakang.
        """
        data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_time = start_date
        while current_time <= end_date:
            hour_of_day = current_time.hour
            
            # Logika Siklus Harian
            # 00-06: Rendah (Tidur)
            # 06-09: Naik (Aktivitas pagi)
            # 09-16: Stabil
            # 17-21: Puncak (Peak Hour)
            # 21-24: Menurun
            
            time_factor = 1.0
            if 0 <= hour_of_day < 6:
                time_factor = 0.3
            elif 17 <= hour_of_day <= 20:
                time_factor = 1.5  # Peak Hour
            else:
                time_factor = 0.8

            for tps in self.tps_locations:
                # Simulasi Level Kepenuhan (0-100)
                # Base load + faktor waktu + random noise
                noise = random.uniform(-10, 20)
                level = int(tps['base_load'] * time_factor + noise)
                
                # Simulasi Pengangkutan (Reset level jika sudah diangkut jam 5 pagi)
                if hour_of_day == 5:
                    level = random.randint(0, 10)
                
                # Cap nilai 0-100
                level = max(0, min(100, level))
                
                # Simulasi Berat (kg) - Korelasi positif dengan level
                # Asumsi max capacity 1 tong besar = 500kg
                berat_kg = level * 5 + random.uniform(-5, 5)
                berat_kg = max(0, round(berat_kg, 2))
                
                # Simulasi Status Tutup
                # Jika > 90% penuh, tutup cenderung terbuka karena sampah meluap
                status_tutup = "Buka" if level > 85 else random.choice(["Tutup", "Tutup", "Tutup", "Buka"]) # 75% chance tutup
                
                # Simulasi Bau
                # Jika Tutup Buka dan Level Tinggi -> Menyengat
                if status_tutup == "Buka" and level > 70:
                    status_bau = "Menyengat"
                else:
                    status_bau = "Normal"

                # Masukkan data record
                data.append({
                    "timestamp": current_time,
                    "id_tps": tps['id'],
                    "nama_lokasi": tps['nama'],
                    "latitude": tps['lat'],
                    "longitude": tps['lon'],
                    "level_kepenuhan": level,
                    "berat_kg": berat_kg,
                    "status_tutup": status_tutup,
                    "status_bau": status_bau
                })
            
            current_time += timedelta(hours=1) # Increment 1 jam
            
        df = pd.DataFrame(data)
        return df

# Helper function untuk caching data agar tidak generate ulang setiap klik
@st.cache_data
def load_data():
    gen = WasteDataGenerator()
    return gen.generate_data(days=7)

# ==============================================================================
# 3. MODUL INTELLIGENT ANALYTICS (SCIKIT-LEARN)
# ==============================================================================

class SmartAnalytics:
    """
    Kelas ini menangani logika 'Cerdas' menggunakan Machine Learning sederhana
    dan algoritma sorting untuk optimasi.
    """
    
    @staticmethod
    def optimize_route(current_data):
        """
        Algoritma Greedy Filter Logic.
        1. Filter TPS dengan kepenuhan > 75% (Status Merah).
        2. Sort berdasarkan kepenuhan tertinggi (Prioritas).
        """
        critical_tps = current_data[current_data['level_kepenuhan'] > 75].copy()
        
        # Sorting prioritas: Level Kepenuhan DESC, Berat DESC
        route_plan = critical_tps.sort_values(
            by=['level_kepenuhan', 'berat_kg'], 
            ascending=[False, False]
        )
        return route_plan

    @staticmethod
    def perform_clustering(df_latest):
        """
        Algoritma K-Means Clustering.
        Mengelompokkan TPS berdasarkan 'level_kepenuhan' dan 'berat_kg'
        menjadi 3 Cluster: Low, Medium, High Priority.
        """
        if len(df_latest) < 3:
            return df_latest # Tidak cukup data untuk clustering
            
        features = df_latest[['level_kepenuhan', 'berat_kg']]
        
        # Scaling data agar balance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # K-Means dengan 3 cluster
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_latest['cluster_id'] = kmeans.fit_predict(scaled_features)
        
        # Mapping cluster ID ke label yang bisa dibaca (Low, Medium, High)
        # Kita perlu cek rata-rata kepenuhan tiap cluster untuk memberi label yang benar
        cluster_means = df_latest.groupby('cluster_id')['level_kepenuhan'].mean().sort_values()
        
        # Mapping logic: mean terendah -> Low, tertinggi -> High
        mapping = {}
        labels = ['Low Priority', 'Medium Priority', 'High Priority']
        for i, cluster_id in enumerate(cluster_means.index):
            mapping[cluster_id] = labels[i]
            
        df_latest['cluster_label'] = df_latest['cluster_id'].map(mapping)
        return df_latest

    @staticmethod
    def predict_waste_generation(df_history):
        """
        Algoritma Simple Linear Regression.
        Memprediksi total berat sampah untuk hari esok.
        """
        # Agregasi data harian
        df_daily = df_history.groupby(df_history['timestamp'].dt.date)['berat_kg'].sum().reset_index()
        df_daily['day_index'] = range(len(df_daily))
        
        X = df_daily[['day_index']]
        y = df_daily['berat_kg']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prediksi hari esok (index terakhir + 1)
        next_day_index = [[len(df_daily)]]
        prediction = model.predict(next_day_index)[0]
        
        return df_daily, prediction, model

# ==============================================================================
# 4. FUNGSI UI & VISUALISASI
# ==============================================================================

def get_color_by_level(level):
    """Mengembalikan warna RGB untuk pydeck berdasarkan level kepenuhan"""
    if level < 50:
        return [25, 135, 84, 160] # Green (RGBA)
    elif level < 80:
        return [255, 193, 7, 160] # Yellow
    else:
        return [220, 53, 69, 200] # Red

def render_sidebar():
    st.sidebar.title("🚛 Smart Waste Mataram")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2362/2362243.png", width=100)
    
    st.sidebar.markdown("---")
    
    # Menu Navigasi Utama
    menu = st.sidebar.radio(
        "Menu Navigasi",
        ["Dashboard Monitoring", "Smart Routing", "Data Analytics", "Laporan Anomali"]
    )
    
    st.sidebar.markdown("---")
    
    # Global Filter
    st.sidebar.subheader("🎛️ Filter Data")
    
    kecamatan_filter = st.sidebar.selectbox(
        "Pilih Kecamatan",
        ["Semua Kecamatan", "Ampenan", "Mataram", "Cakranegara", "Sekarbela", "Selaparang", "Sandubaya"]
    )
    
    st.sidebar.info(
        """
        **Status Sistem:** Online 🟢
        \n**Update Terakhir:** """ + datetime.now().strftime("%H:%M:%S")
    )
    
    return menu, kecamatan_filter

def render_dashboard_page(df, kecamatan_filter):
    st.title("🏙️ Dashboard Monitoring Real-time")
    st.markdown("Pantauan kondisi TPS di seluruh Kota Mataram secara real-time.")
    
    # 1. Filter Data Terakhir (Snapshot saat ini)
    latest_timestamp = df['timestamp'].max()
    df_latest = df[df['timestamp'] == latest_timestamp].copy()
    
    if kecamatan_filter != "Semua Kecamatan":
        df_latest = df_latest[df_latest['nama_lokasi'].str.contains(kecamatan_filter)]
    
    # 2. Key Metrics (Scorecard)
    col1, col2, col3, col4 = st.columns(4)
    
    total_sampah = df_latest['berat_kg'].sum() / 1000 # dalam Ton
    avg_kepenuhan = df_latest['level_kepenuhan'].mean()
    tps_kritis = len(df_latest[df_latest['level_kepenuhan'] > 80])
    bau_menyengat = len(df_latest[df_latest['status_bau'] == 'Menyengat'])
    
    col1.metric("Total Sampah (Ton)", f"{total_sampah:.2f}", "+0.5%")
    col2.metric("Rata-rata Penuh", f"{avg_kepenuhan:.1f}%", f"{avg_kepenuhan-50:.1f}%")
    col3.metric("TPS Status Merah", f"{tps_kritis} Lokasi", "Perlu Tindakan", delta_color="inverse")
    col4.metric("Laporan Bau", f"{bau_menyengat} Titik", "Lingkungan", delta_color="inverse")
    
    st.markdown("---")
    
    # 3. Peta Interaktif (PyDeck)
    st.subheader("📍 Peta Sebaran TPS")
    
    # Persiapkan data warna untuk peta
    df_latest['color'] = df_latest['level_kepenuhan'].apply(get_color_by_level)
    
    # Definisi View State (Fokus ke Mataram)
    view_state = pdk.ViewState(
        latitude=-8.58,
        longitude=116.11,
        zoom=12,
        pitch=45,
    )
    
    # Layer: Column Layer (Tinggi batang = Level Kepenuhan)
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df_latest,
        get_position=["longitude", "latitude"],
        get_elevation="level_kepenuhan",
        elevation_scale=10,
        radius=100,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    
    # Tooltip
    tooltip = {
        "html": "<b>{nama_lokasi}</b><br/>Level: {level_kepenuhan}%<br/>Berat: {berat_kg} kg<br/>Status: {status_tutup}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    r = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    st.pydeck_chart(r)
    
    # 4. Tabel Monitoring Detail
    st.subheader("📋 Detail Status TPS")
    
    # Styling DataFrame dengan Pandas Styler
    def highlight_status(val):
        if val > 80: return 'background-color: #ffcccc; color: red; font-weight: bold'
        elif val > 50: return 'background-color: #ffffe0; color: orange'
        else: return 'background-color: #ccffcc; color: green'
        
    st.dataframe(
        df_latest[['id_tps', 'nama_lokasi', 'level_kepenuhan', 'berat_kg', 'status_tutup', 'status_bau']]
        .sort_values(by='level_kepenuhan', ascending=False)
        .style.applymap(highlight_status, subset=['level_kepenuhan'])
        .format({"level_kepenuhan": "{:.0f}%", "berat_kg": "{:.1f} kg"}),
        use_container_width=True
    )

def render_routing_page(df):
    st.title("🚚 Smart Routing Optimization")
    st.markdown("Rekomendasi rute pengangkutan berdasarkan prioritas kepenuhan sampah.")
    
    # Ambil data snapshot terakhir
    latest_timestamp = df['timestamp'].max()
    df_latest = df[df['timestamp'] == latest_timestamp].copy()
    
    # Jalankan Algoritma Optimasi
    analytics = SmartAnalytics()
    route_plan = analytics.optimize_route(df_latest)
    
    col_kiri, col_kanan = st.columns([1, 2])
    
    with col_kiri:
        st.success(f"**Rekomendasi AI:** Ditemukan **{len(route_plan)} TPS Kritis** yang harus segera diangkut.")
        
        st.markdown("### 📝 Urutan Penjemputan")
        if route_plan.empty:
            st.info("Tidak ada TPS kritis saat ini. Armada bisa standby.")
        else:
            for idx, row in route_plan.reset_index().iterrows():
                st.write(f"**{idx+1}. {row['nama_lokasi']}**")
                st.caption(f"ID: {row['id_tps']} | Penuh: {row['level_kepenuhan']}% | Beban: {row['berat_kg']}kg")
                if st.checkbox(f"Tandai Selesai (Stop {idx+1})", key=f"chk_{row['id_tps']}"):
                    st.write("✅ *Terkonfirmasi*")
                st.markdown("---")

    with col_kanan:
        st.markdown("### 🗺️ Visualisasi Rute")
        
        if not route_plan.empty:
            # Membuat Garis Rute (Path Layer)
            # Kita hubungkan titik secara berurutan
            path_data = []
            coords = route_plan[['longitude', 'latitude']].values.tolist()
            
            # Tambahkan TPA Kebon Kongok (Lokasi fiktif TPA di selatan Mataram)
            tpa_coords = [116.15, -8.65] 
            coords.append(tpa_coords)
            
            # Data untuk PathLayer Pydeck butuh format spesifik
            path_data = [{"path": coords, "name": "Rute Armada 1"}]
            
            view_state = pdk.ViewState(latitude=-8.58, longitude=116.11, zoom=11)
            
            # Layer Titik TPS
            layer_points = pdk.Layer(
                "ScatterplotLayer",
                data=route_plan,
                get_position=["longitude", "latitude"],
                get_color=[255, 0, 0, 200],
                get_radius=200,
                pickable=True
            )
            
            # Layer Garis Rute
            layer_path = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[0, 0, 255, 150],
                width_scale=20,
                width_min_pixels=3,
                pickable=True
            )
            
            r = pdk.Deck(layers=[layer_path, layer_points], initial_view_state=view_state, map_style="mapbox://styles/mapbox/streets-v11")
            st.pydeck_chart(r)
            st.caption("Garis Biru: Rute yang disarankan. Titik Merah: TPS Target. Ujung akhir: TPA.")
        else:
            st.image("https://img.freepik.com/free-vector/garbage-truck-concept-illustration_114360-12850.jpg", width=400)
            st.write("Semua TPS dalam kondisi aman.")

def render_analytics_page(df):
    st.title("📈 Data Analytics & Prediction")
    st.markdown("Analisis mendalam tren sampah dan prediksi masa depan untuk perencanaan kebijakan.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Clustering Wilayah", "🔮 Prediksi Timbulan", "📉 Tren Historis"])
    
    # Snapshot data terakhir untuk clustering
    latest_timestamp = df['timestamp'].max()
    df_latest = df[df['timestamp'] == latest_timestamp].copy()
    
    with tab1:
        st.subheader("Segmentasi Karakteristik Wilayah (K-Means)")
        st.markdown("Algoritma AI mengelompokkan TPS berdasarkan pola kepenuhan dan berat.")
        
        # Jalankan Clustering
        analytics = SmartAnalytics()
        df_clustered = analytics.perform_clustering(df_latest)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.scatter_mapbox(
                df_clustered, 
                lat="latitude", 
                lon="longitude", 
                color="cluster_label",
                size="berat_kg",
                hover_name="nama_lokasi",
                color_discrete_map={
                    "High Priority": "red",
                    "Medium Priority": "orange",
                    "Low Priority": "green"
                },
                zoom=12,
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Insight Cluster")
            avg_stats = df_clustered.groupby('cluster_label')[['level_kepenuhan', 'berat_kg']].mean().reset_index()
            st.dataframe(avg_stats.style.format("{:.1f}"), use_container_width=True)
            st.info("Area 'High Priority' membutuhkan frekuensi pengangkutan 2x sehari.")

    with tab2:
        st.subheader("Prediksi Total Berat Sampah (Linear Regression)")
        
        # Jalankan Prediksi
        df_daily, pred_val, model = SmartAnalytics.predict_waste_generation(df)
        
        col_metric_1, col_metric_2 = st.columns(2)
        col_metric_1.metric("Total Sampah Kemarin", f"{df_daily['berat_kg'].iloc[-1]:,.0f} kg")
        col_metric_2.metric("Prediksi Besok", f"{pred_val:,.0f} kg", f"{(pred_val - df_daily['berat_kg'].iloc[-1]):.0f} kg")
        
        # Plot Grafik Regresi
        fig = go.Figure()
        
        # Data Aktual
        fig.add_trace(go.Scatter(
            x=df_daily['timestamp'], 
            y=df_daily['berat_kg'],
            mode='lines+markers',
            name='Data Aktual',
            line=dict(color='#0D6EFD', width=3)
        ))
        
        # Garis Tren (Regresi)
        y_trend = model.predict(df_daily[['day_index']])
        fig.add_trace(go.Scatter(
            x=df_daily['timestamp'],
            y=y_trend,
            mode='lines',
            name='Trend Line (AI)',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Tren Peningkatan Sampah Harian",
            yaxis_title="Total Berat (kg)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f} (Model Regresi Sederhana)")

    with tab3:
        st.subheader("Tren Jam Sibuk (Peak Hour Analysis)")
        
        # Ekstrak Jam
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['level_kepenuhan'].mean().reset_index()
        
        fig_bar = px.bar(
            hourly_avg, 
            x='hour', 
            y='level_kepenuhan',
            labels={'hour': 'Jam', 'level_kepenuhan': 'Rata-rata Kepenuhan (%)'},
            color='level_kepenuhan',
            color_continuous_scale='RdYlGn_r' # Red Yellow Green (Reversed)
        )
        fig_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.write("Insight: Grafik di atas menunjukkan rata-rata kepenuhan tong sampah berdasarkan jam. Perhatikan lonjakan di sore hari.")

def render_anomaly_report(df):
    st.title("⚠️ Laporan Anomali & Kerusakan")
    
    # Filter kondisi bau menyengat atau tutup rusak
    df_bau = df[df['status_bau'] == 'Menyengat']
    df_tutup = df[df['status_tutup'] == 'Buka'] # Asumsi buka terus = anomali jika hujan
    
    st.error(f"Terdeteksi **{len(df_bau)} insiden** bau menyengat dalam 7 hari terakhir.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top 5 Lokasi Paling Bau")
        top_bau = df_bau['nama_lokasi'].value_counts().head(5)
        st.bar_chart(top_bau)
        
    with col2:
        st.markdown("### Distribusi Status Tutup")
        pie_data = df['status_tutup'].value_counts()
        fig = px.pie(values=pie_data.values, names=pie_data.index, hole=0.4, color_discrete_sequence=['#198754', '#dc3545'])
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. MAIN EXECUTION FLOW
# ==============================================================================

def main():
    # Load Data (Cached)
    df = load_data()
    
    # Render Sidebar & Get Selection
    selected_menu, kecamatan_filter = render_sidebar()
    
    # Routing Halaman
    if selected_menu == "Dashboard Monitoring":
        render_dashboard_page(df, kecamatan_filter)
        
    elif selected_menu == "Smart Routing":
        render_routing_page(df)
        
    elif selected_menu == "Data Analytics":
        render_analytics_page(df)
        
    elif selected_menu == "Laporan Anomali":
        render_anomaly_report(df)

if __name__ == "__main__":
    main()