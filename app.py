import streamlit as st
import ee
import folium
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from google.oauth2 import service_account

st.set_page_config(page_title="Dashboard Kualitas Udara Kaltim", layout="wide")
st.title("🛰️ Dashboard Estimasi PM2.5 & Indikator Paparan")
st.markdown("**Lokasi:** Kalimantan Timur | **Pemodelan:** Random Forest Machine Learning | **Periode:** Agustus 2026")
st.markdown("---")
st.info(
    "Prototipe akademik. Keakuratan estimasi PM2.5 serta satuan statistik paparan "
    "belum tervalidasi secara independen; jangan gunakan sebagai informasi "
    "kualitas udara real-time atau dasar keputusan kesehatan."
)

# Pada Cloud gunakan secret service-account; pada Colab boleh memakai
# autentikasi Earth Engine pengguna yang telah diaktifkan.
GEE_PROJECT = "shintya-project"
try:
    secret_json = st.secrets.get("gee_service_account_json")
except (FileNotFoundError, KeyError):
    secret_json = None

try:
    if secret_json:
        credentials_info = json.loads(secret_json)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=[
                "https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )
        ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    else:
        ee.Initialize(project=GEE_PROJECT)
except Exception:
    st.error(
        "Google Earth Engine belum dapat diakses. Di Streamlit Cloud, "
        "atur secret `gee_service_account_json`, beri akses Earth Engine "
        "pada service account, dan berikan akses baca ke asset ROI. "
        "Jangan mengunggah kunci JSON ke GitHub."
    )
    st.stop()

# Batas wilayah dan periode analisis
roi = ee.FeatureCollection('projects/shintya-project/assets/KalimantanTimur')
start_date = '2026-08-01'
end_date = '2026-09-01'  # filterDate: tanggal akhir tidak disertakan

col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("🗺️ Peta Interaktif Sebaran PM2.5 & Parameter")
    map_dash = folium.Map(location=[-0.5, 117.0], zoom_start=6)

    # Fungsi untuk menampilkan raster Earth Engine
    def add_ee_layer(self, ee_image_object, vis_params, name, show=True):
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data &copy; Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=show
        ).add_to(self)

    folium.Map.add_ee_layer = add_ee_layer

    # Kotak sederhana untuk memfilter koleksi Sentinel-5P
    bbox_dash = ee.Geometry.Rectangle(
        [113.0, -3.0, 120.0, 3.0],
        geodesic=False
    )

    def raster_dashboard(collection_id, band):
        return (
            ee.ImageCollection(collection_id)
            .filterDate(start_date, end_date)
            .filterBounds(bbox_dash)
            .select(band)
            .mean()
            .clipToCollection(roi)
        )

    # A. DATA METEOROLOGI
    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .filterDate(start_date, end_date)
        .select([
            'temperature_2m',
            'u_component_of_wind_10m'
        ])
        .mean()
        .clipToCollection(roi)
    )

    map_dash.add_ee_layer(
        era5.select('u_component_of_wind_10m'),
        {
            'min': -3.0,
            'max': 3.0,
            'palette': ['blue', 'white', 'red']
        },
        'Wind_U',
        show=False
    )

    map_dash.add_ee_layer(
        era5.select('temperature_2m'),
        {
            'min': 295,
            'max': 305,
            'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
        },
        'Temperature_2m',
        show=False
    )

    # B. DATA KUALITAS UDARA
    palette_s5p = [
        'black', 'blue', 'purple',
        'cyan', 'green', 'yellow', 'red'
    ]

    no2 = raster_dashboard(
        'COPERNICUS/S5P/NRTI/L3_NO2',
        'NO2_column_number_density'
    )

    map_dash.add_ee_layer(
        no2,
        {'min': 0, 'max': 0.0002, 'palette': palette_s5p},
        'NO2',
        show=False
    )

    hcho = raster_dashboard(
        'COPERNICUS/S5P/NRTI/L3_HCHO',
        'tropospheric_HCHO_column_number_density'
    )

    map_dash.add_ee_layer(
        hcho,
        {'min': 0, 'max': 0.0005, 'palette': palette_s5p},
        'HCHO',
        show=False
    )

    ai_raw = raster_dashboard(
        'COPERNICUS/S5P/NRTI/L3_AER_AI',
        'absorbing_aerosol_index'
    )

    map_dash.add_ee_layer(
        ai_raw,
        {'min': -1.5, 'max': 1.5, 'palette': palette_s5p},
        'Aerosol Index',
        show=False
    )

    # C. PREDIKSI PM2.5 MENGGUNAKAN RANDOM FOREST
    with st.spinner("Memuat Data Latih dan Memproses Random Forest..."):

        # Nama band disamakan dengan kolom data training.
        # Suhu untuk input model diubah dari Kelvin ke Celsius.
        predictors_renamed = (
            ai_raw.rename('Aerosol_Index')
            .addBands(
                era5.select('temperature_2m')
                .subtract(273.15)
                .rename('Temperature_2m')
            )
            .addBands(
                era5.select('u_component_of_wind_10m')
                .rename('Wind_U')
            )
            .addBands(no2.rename('NO2_Density'))
            .addBands(hcho.rename('HCHO_Density'))
        )

        # Membaca data training hasil simulasi dari Colab
        df_train = pd.read_csv(Path(__file__).resolve().parent / 'training_data_pm25.csv')

        # Mengubah tabel training menjadi FeatureCollection GEE
        features = [
            'Aerosol_Index',
            'Temperature_2m',
            'Wind_U',
            'NO2_Density',
            'HCHO_Density'
        ]

        ee_features = []

        for index, row in df_train.iterrows():
            feat = ee.Feature(
                ee.Geometry.Point([
                    row['longitude'],
                    row['latitude']
                ]),
                {
                    'Aerosol_Index': row['Aerosol_Index'],
                    'Temperature_2m': row['Temperature_2m'],
                    'Wind_U': row['Wind_U'],
                    'NO2_Density': row['NO2_Density'],
                    'HCHO_Density': row['HCHO_Density'],
                    'Actual_PM25': row['Actual_PM25']
                }
            )
            ee_features.append(feat)

        fc_training = ee.FeatureCollection(ee_features)

        # Melatih model regresi Random Forest
        classifier = (
            ee.Classifier.smileRandomForest(10)
            .setOutputMode('REGRESSION')
            .train(
                features=fc_training,
                classProperty='Actual_PM25',
                inputProperties=features
            )
        )

        # Raster input sudah dipotong menggunakan clipToCollection.
        predicted_pm25 = predictors_renamed.classify(classifier)

        map_dash.add_ee_layer(
            predicted_pm25,
            {
                'min': 0,
                'max': 100,
                'palette': [
                    '008000', 'FFFF00', 'FFA500',
                    'FF0000', '800080'
                ]
            },
            'Prediksi PM2.5',
            show=True
        )

        # Ambang 55.5 digunakan hanya sebagai skenario eksploratif.
        # Tidak boleh disetarakan dengan baku mutu bulanan.
        # Zona dengan nilai PM2.5 di atas ambang yang digunakan
        danger_zone = predicted_pm25.gt(55.5)
        danger_masked = danger_zone.updateMask(danger_zone.eq(1))

        map_dash.add_ee_layer(
            danger_masked,
            {'min': 0, 'max': 1, 'palette': ['red']},
            'Zona di Atas 55.5 µg/m³ (indikatif)',
            show=True
        )

    # D. GARIS BATAS ADMINISTRASI KALTIM
    outline = (
        ee.Image(0).byte()
        .paint(roi, color=1, width=2)
        .selfMask()
    )

    map_dash.add_ee_layer(
        outline,
        {'min': 1, 'max': 1, 'palette': ['black']},
        'Batas Administrasi Kaltim',
        show=True
    )

    map_dash.add_child(folium.LayerControl())

    # Menampilkan peta interaktif
    components.html(map_dash._repr_html_(), height=500)

    # E. LEGENDA PARAMETER
    with st.expander("📖 Lihat Legenda Parameter (Warna)", expanded=False):
        st.markdown("""
<div style="display:flex;gap:20px;flex-wrap:wrap;">
<div style="flex:1;min-width:200px;">
<b>Prediksi PM2.5 (µg/m³)</b>
<div style="background:linear-gradient(to right,#008000,#FFFF00,#FFA500,#FF0000,#800080);height:12px;border-radius:5px;margin-top:5px;"></div>
<div style="display:flex;justify-content:space-between;font-size:12px;"><span>0</span><span>100</span></div>
</div>
<div style="flex:1;min-width:200px;">
<b>Aerosol Index (AI)</b>
<div style="background:linear-gradient(to right,black,blue,purple,cyan,green,yellow,red);height:12px;border-radius:5px;margin-top:5px;"></div>
<div style="display:flex;justify-content:space-between;font-size:12px;"><span>-1.5</span><span>+1.5</span></div>
</div>
<div style="flex:1;min-width:200px;">
<b>Suhu Udara 2 m (Kelvin)</b>
<div style="background:linear-gradient(to right,blue,cyan,green,yellow,red);height:12px;border-radius:5px;margin-top:5px;"></div>
<div style="display:flex;justify-content:space-between;font-size:12px;"><span>295 K</span><span>305 K</span></div>
</div>
<div style="flex:1;min-width:200px;">
<b>Komponen Angin Zonal U (m/s)</b>
<div style="background:linear-gradient(to right,blue,white,red);height:12px;border-radius:5px;margin-top:5px;"></div>
<div style="display:flex;justify-content:space-between;font-size:12px;"><span>-3.0</span><span>+3.0</span></div>
</div>
<div style="flex:1;min-width:200px;">
<b>NO2 / HCHO Density</b>
<div style="background:linear-gradient(to right,black,blue,purple,cyan,green,yellow,red);height:12px;border-radius:5px;margin-top:5px;"></div>
<div style="display:flex;justify-content:space-between;font-size:12px;"><span>0.0 (Rendah)</span><span>Tinggi</span></div>
</div>
</div>
""", unsafe_allow_html=True)

# F. STATISTIK DAMPAK (CSV MASIH PERLU VALIDASI SATUAN)
with col2:
    st.subheader("📊 Indikator Area Terpapar")
    st.warning(
        "Nilai di bawah berasal dari CSV hasil pengolahan sebelumnya. "
        "Satuan dan metode agregasinya belum diverifikasi, sehingga "
        "belum boleh disebut sebagai jumlah jiwa terpapar."
    )
    try:
        df_impact = pd.read_csv(Path(__file__).resolve().parent / 'data_dampak_kaltim.csv')
        required = {'Kabupaten', 'Populasi_Terdampak'}
        if not required.issubset(df_impact.columns):
            st.error('Kolom wajib pada data_dampak_kaltim.csv tidak tersedia.')
            st.stop()

        # Hanya 10 wilayah administratif Kalimantan Timur.
        kaltim = {
            'Berau', 'Kutai Barat', 'Kutai Kartanegara', 'Kutai Timur',
            'Mahakam Ulu', 'Paser', 'Penajam Paser Utara',
            'Kota Balikpapan', 'Kota Bontang', 'Kota Samarinda',
        }
        # Nama 'Pasir' adalah nama lama Kabupaten Paser.
        df_impact['Kabupaten'] = df_impact['Kabupaten'].replace({'Pasir': 'Paser'})
        df_impact = df_impact[df_impact['Kabupaten'].isin(kaltim)].copy()
        expected = {
            'Berau', 'Kutai Barat', 'Kutai Kartanegara', 'Kutai Timur',
            'Mahakam Ulu', 'Paser', 'Penajam Paser Utara',
            'Kota Balikpapan', 'Kota Bontang', 'Kota Samarinda',
        }
        missing = sorted(expected - set(df_impact['Kabupaten']))
        if missing:
            st.caption('Belum ada data untuk: ' + ', '.join(missing))
        if df_impact.empty:
            st.warning('Belum ada data statistik dampak untuk Kalimantan Timur.')
        else:
            df_impact['Populasi_Terdampak'] = pd.to_numeric(
                df_impact['Populasi_Terdampak'], errors='coerce'
            )
            df_impact = df_impact.dropna(subset=['Populasi_Terdampak'])
            df_impact = df_impact.sort_values('Populasi_Terdampak', ascending=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(
                df_impact['Kabupaten'], df_impact['Populasi_Terdampak'],
                color='crimson'
            )
            ax.set_xlabel('Nilai pada CSV (satuan belum terverifikasi)')
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            with st.expander('Lihat tabel sumber grafik'):
                st.dataframe(df_impact, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.warning('File data_dampak_kaltim.csv belum tersedia di repository.')

# =====================================================
# WEBGIS FOOTER - GEOAIR INTELLIGENCE
# =====================================================

st.html('''
<style>
.geoair-footer {
    background: linear-gradient(120deg, #101e30, #19384b);
    border: 1px solid #2c4d61;
    border-radius: 16px;
    overflow: hidden;
    margin-top: 28px;
    font-family: Arial, sans-serif;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.geoair-accent {
    height: 4px;
    background: linear-gradient(
        90deg,
        #25ba83 0%,
        #f5db41 28%,
        #f18c38 50%,
        #dc3c54 75%,
        #803cb8 100%
    );
}
.geoair-inner {
    padding: 24px 28px 18px;
}
.geoair-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 25px;
    flex-wrap: wrap;
}
.geoair-title {
    font-size: 23px;
    font-weight: 750;
    color: #f2f8fc;
    margin: 0 0 9px;
    line-height: 1.3;
}
.geoair-title span {
    color: #65dfd9;
}
.geoair-desc {
    font-size: 13px;
    color: #bed1df;
    margin: 0 0 7px;
}
.geoair-motto {
    font-size: 12px;
    color: #90aebf;
    margin: 0;
}
.geoair-meta {
    font-size: 11px;
    font-weight: bold;
    color: #69d9d3;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.geoair-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}
.geoair-tag {
    background: #233e52;
    border: 1px solid #34566a;
    color: #d4e8f2;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}
.geoair-bottom {
    border-top: 1px solid #345064;
    margin-top: 23px;
    padding-top: 15px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    color: #a4b9c9;
    font-size: 11px;
}
@media (max-width: 650px) {
    .geoair-inner { padding: 20px 16px; }
    .geoair-title { font-size: 20px; }
    .geoair-top { gap: 18px; }
}
</style>

<footer class="geoair-footer">
    <div class="geoair-accent"></div>

    <div class="geoair-inner">
        <div class="geoair-top">

            <div>
                <h2 class="geoair-title">
                    🌍 GeoAir <span>Intelligence</span>
                </h2>

                <p class="geoair-desc">
                    Geospatial PM₂.₅ Prediction
                    &amp; Population Exposure Analysis
                </p>

                <p class="geoair-motto">
                    Mapping Air Pollution.
                    Enabling Smarter Decisions.
                </p>
            </div>

            <div>
                <div class="geoair-meta">
                    EAST KALIMANTAN | AUGUST 2026
                </div>

                <div class="geoair-tags">
                    <span class="geoair-tag">
                        Google Earth Engine
                    </span>
                    <span class="geoair-tag">
                        Python
                    </span>
                    <span class="geoair-tag">
                        Streamlit
                    </span>
                    <span class="geoair-tag">
                        Random Forest
                    </span>
                </div>
            </div>

        </div>

        <div class="geoair-bottom">
            <span>
                © 2026 GeoAir Intelligence |
                Developed by Shintya Defita
            </span>

            <span>
                🛰 GIS · Remote Sensing · Spatial Analytics
            </span>
        </div>
    </div>
</footer>
''')

