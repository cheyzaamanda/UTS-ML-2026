# Reproduksi Kode UTS Machine Learning 2026 (Task 1 Regresi & Task 2 Klasifikasi)

Folder `kode/` ini berisi seluruh modul pipeline *classical machine learning* untuk mereproduksi angka evaluasi di laporan serta menghasilkan file prediksi resmi sesuai format autograder UTS:

- **`prediksi_regresi.csv`** — Header persis `id_paket,p50,p90` (60.000 baris data uji, dengan jaminan `p90 >= p50` dan `p50 > 0`).
- **`prediksi_klasifikasi.csv`** — Header persis `id_transaksi,prob,tandai` (106.674 baris data uji, dengan `0 <= prob <= 1` dan `tandai` bernilai `0` atau `1` berdasarkan aturan biaya review **Rp 25.000**).

---

## 1. Struktur File

- `kode/config.py` — Konfigurasi terpusat (`SEED = 42`, `REVIEW_COST_IDR = 25_000`, resolusi `DATA_DIR` / `UTS_DATA_DIR`, pencatatan versi library, dan utilitas I/O).
- `kode/regresi.py` — Pipeline lengkap **Task 1 (Estimasi Waktu Pengiriman / `durasi_jam`)**:
  - **(a) Audit Data & Leakage:** Deteksi kebocoran pasca-pickup (`jumlah_scan`), error satuan 60x pada `durasi_jam > 1000`, sentinel `jarak_km` (`0` / `9999`), missingness MNAR `berat_kg` pada layanan `kargo`, dan pergeseran fitur `f30`/`f35`.
  - **(b) Validasi Berbasis Waktu vs Random CV:** *Expanding-window time-based validation* dibandingkan dengan *5-fold Random CV*.
  - **(c–d) Pemodelan Kuantil & Kategori:** Baseline kuantil + model *gradient boosting* (`HistGradientBoostingRegressor`, `LightGBM`, `XGBoost` / linear target-transformed) dengan penanganan *high-cardinality* (`kota_tujuan`, `hub_asal`) dan kategori *unseen* di data uji (`mitra_kurir = K13`).
  - **(e) Ketidakpastian & Kalibrasi:** Evaluasi *pinball loss* (`q = 0.5` dan `q = 0.9`), *P90 coverage*, serta kalibrasi *split conformal prediction*.
  - **(f) Analisis Error & Interpretabilitas:** Rincian error per `layanan`, `pulau_tujuan`, dan `mitra_kurir`, dilengkapi analisis **SHAP** dan **Partial Dependence (PDP)**.
- `kode/klasifikasi.py` — Pipeline lengkap **Task 2 (Deteksi Fraud / `is_fraud`)**:
  - **(a) Audit Label & Fitur:** Deteksi kebocoran `flag_review_manual`, drift monontonik `v15`, duplikasi `v26`, anomali `umur_merchant_hari <= 0`, dan *cold-start nulls*.
  - **(b) Validasi Waktu & Label Delay:** Penanganan *right-censoring* akhir Agustus 2025 (label fraud yang belum sempat terkonfirmasi saat data ditarik 31 Agustus 2025) dengan *Out-of-Time (OOT) validation*.
  - **(c) Fitur Perilaku Kausal:** Rekayasa fitur riwayat pengguna dan merchant secara kausal (*strictly prior history*, tanpa kebocoran masa depan).
  - **(d–e) Imbalance & Pergeseran Pola:** Perbandingan pendekatan *class imbalance* (`unweighted`, `class_weight`, *undersampling Ensemble*, *recency weighting*) dan adaptasi terhadap lonjakan pola fraud `kanal = qris` & `kategori_merchant = jasa_lainnya`.
  - **(f) Kalibrasi Probabilitas & Keputusan Biaya Rp 25.000:** Kalibrasi probabilitas (OOF Brier & PR-AUC) dan optimasi aturan keputusan berbasis biaya (`cost = sum(nominal missed fraud) + 25000 * jumlah_ditandai`) dibandingkan strategi *flag all* dan *flag none*.
  - **(g) Analisis Error & Interpretabilitas:** Rincian error dan PR-AUC (`average_precision`) berdasarkan `kanal`, `kategori_merchant`, dan pola fraud, disertai nilai **SHAP**.
- `kode/run_all.py` — Skrip **satu perintah** untuk menjalankan seluruh pipeline Task 1 dan Task 2 dari terminal.
- `UTS_ML_Regresi_VSCode.ipynb` & `UTS_ML_Klasifikasi_VSCode.ipynb` — Notebook interaktif VS Code yang terintegrasi langsung dengan modul `kode.regresi`, `kode.klasifikasi`, dan `kode.config`.

---

## 2. Lingkungan, Seed, & Versi Library

- **Random Seed Global:** `SEED = 42`
- **Lokasi Data Default (`DATA_DIR`):** `~/Downloads/data_uts_ml` (dapat diubah melalui argumen `--data-dir` atau environment variable `UTS_DATA_DIR`).
- **Instalasi/Sinkronisasi Paket di `.venv`:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_ml.txt
```

Versi paket utama yang di-pin pada `requirements_ml.txt` (hanya *classical machine learning* sesuai ketentuan UTS):
- `numpy==2.4.3`, `pandas==2.3.3`, `scipy==1.17.1`, `scikit-learn==1.8.0`
- `lightgbm==4.6.0`, `xgboost==3.2.0`, `catboost==1.2.10`, `statsmodels==0.14.6`
- `shap==0.51.0`, `imbalanced-learn==0.14.1`, `optuna==4.8.0`

---

## 3. Satu Perintah Reproduksi (One-Command Reproduction)

Jalankan perintah berikut dari **root repository** menggunakan interpreter `.venv` untuk mereproduksi seluruh metrik evaluasi sekaligus menghasilkan `prediksi_regresi.csv` dan `prediksi_klasifikasi.csv` di root folder:

```powershell
.\.venv\Scripts\python.exe kode/run_all.py --data-dir "C:\Users\rafai\Downloads\data_uts_ml" --artifacts-dir output/artifacts
```

### Opsi Tambahan CLI (`kode/run_all.py`)

- **Verifikasi Cepat (*Smoke Test* / Dry-Run):**
  ```powershell
  .\.venv\Scripts\python.exe kode/run_all.py --smoke-test
  ```
- **Menjalankan Satu Tugas Saja (`regresi` atau `klasifikasi`):**
  ```powershell
  .\.venv\Scripts\python.exe kode/run_all.py --only regresi
  .\.venv\Scripts\python.exe kode/run_all.py --only klasifikasi
  ```
- **Menjalankan lewat Notebook Interaktif di VS Code:**
  Buka `UTS_ML_Regresi_VSCode.ipynb` atau `UTS_ML_Klasifikasi_VSCode.ipynb` di VS Code, pilih kernel Python dari `.venv`, lalu klik **Run All**.
