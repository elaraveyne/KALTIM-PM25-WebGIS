# GeoAir Intelligence – Kalimantan Timur (Agustus 2026)

Prototipe WebGIS estimasi PM2.5 menggunakan Google Earth Engine, Random Forest, Folium, dan Streamlit.

## Isi paket
- `app.py`: aplikasi web, berisi peta dan grafik.
- `training_data_pm25.csv`: data target/prediktor dari Colab.
- `data_dampak_kaltim.csv`: nilai agregat dampak **belum tervalidasi satuannya**.
- `requirements.txt`: dependensi saat deployment.
- `.gitignore`: perlindungan agar kunci akun layanan tidak terunggah.

## Deploy GitHub -> Streamlit Community Cloud
1. Buat repository GitHub (`KALTIM-PM25-WebGIS`) dan unggah semua file dari folder paket.
2. Buka https://share.streamlit.io dan login memakai GitHub. Klik **Create app**; pilih repo dan file utama `app.py`.
3. Di **Advanced settings → Secrets**, simpan JSON service account dalam format TOML:

   ```toml
   gee_service_account_json = '''
   {"type":"service_account","project_id":"...","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"...","token_uri":"https://oauth2.googleapis.com/token"}
   '''
   ```

   **Ganti seluruh contoh JSON** dengan isi penuh kunci JSON resmi dari Google Cloud. Jangan tempel kredensial di chat atau unggah file kunci ke GitHub.
4. Google Cloud project `shintya-project` harus mengaktifkan Earth Engine API serta mempunyai izin Earth Engine yang sesuai. Service account perlu izin untuk menggunakan Earth Engine dan **akses baca** pada asset `projects/shintya-project/assets/KalimantanTimur`.
5. Klik **Deploy**, tunggu build selesai, dan uji apakah peta + grafik dapat dimuat.

## Catatan ilmiah sebelum publikasi final
- Kode saat ini melatih ulang Random Forest dari `training_data_pm25.csv`, yang harus diverifikasi asal nilai `Actual_PM25`-nya (observasi/produk model/simulasi).
- Filter tanggal end-exclusive disetel sampai `2026-09-01` agar mencakup 31 Agustus.
- `data_dampak_kaltim.csv` aslinya masih mencampur beberapa daerah Kalimantan Utara, maka app hanya menampilkan baris untuk Kaltim. Kabupaten Mahakam Ulu belum ada; **perhitungan sumber perlu diperbaiki**, bukan hanya grafik.
- Kolom `Populasi_Terdampak` belum jelas satuannya. Dashboard memberi disclaimer dan tidak menyebut angkanya sebagai jumlah jiwa.
- Kriteria 55.5 µg/m³ adalah **skenario ambang**, jangan disebut ambang aman/tidak aman untuk rerata bulanan tanpa justifikasi.
- `roi` tergantung asset privat Earth Engine. Jika service account tidak diberi izin, website gagal memuat peta.
- Jumlah pohon (`10`) mengikuti model asli. Akurasi, sumber target, desain validasi spasial, serta reliabilitas hasil perlu diuji sebelum publikasi ilmiah.

## Referensi teknis
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- https://developers.google.com/earth-engine/guides/service_account

Sumber awal: file `app.py` dan dua CSV dari mini project Kalimantan Timur. Nama pengembang dipertahankan sesuai footer asli.
