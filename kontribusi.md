# Kontribusi Kelompok – UTS Machine Learning 2026

* **Mata Kuliah:** Machine Learning (S1 Teknik Komputer, Fakultas Teknik Elektro, Universitas Telkom)
* **Repositori Utama:** https://github.com/cheyzaamanda/UTS-ML-2026

---

## 1. Anggota Kelompok dan Pembagian Peran

### A. Cheyza Amanda Syafira (NIM: 10103230081 | GitHub: [@cheyzaamanda](https://github.com/cheyzaamanda))
* Inisialisasi repositori utama kelompok (`UTS-ML-2026`).
* Pengerjaan **Task 3 – Clustering** (EDA & audit data kompositional/skewness, transformasi CLR, seleksi fitur 3 tahap dengan $\eta^2$ & uji ablasi, perbandingan K-Means/GMM/Agglomerative, uji stabilitas ARI, deteksi anomali Isolation Forest, dan pembuatan `cluster.csv`).
* Penyusunan bagian Task 3 dan Appendix terkait pada `laporan.pdf` serta peninjauan & *merge* Pull Request ke branch `main`.

### B. Rafaini Zahra Syazwina (NIM: 101032300106 | GitHub: [@zeezaayy](https://github.com/zeezaayy))
* Fork repositori dan kontribusi melalui alur Pull Request.
* Pengerjaan **Task 1 – Regression** (audit *data leakage* `jumlah_scan` & koreksi satuan target, validasi *expanding-window Time-Based CV*, pemodelan regresi kuantil LightGBM, kalibrasi *Split Conformal*, analisis SHAP/PDP, dan pembuatan `prediksi_regresi.csv`).
* Pengerjaan **Task 2 – Classification** (audit *right-censoring* & kebocoran fitur, rekayasa fitur perilaku kausal, perbandingan 6 strategi *class imbalance*, kalibrasi probabilitas *isotonic*, aturan keputusan berbasis biaya, dan pembuatan `prediksi_klasifikasi.csv`).
* Penyusunan bagian Task 1, Task 2, Kesimpulan, serta finalisasi `laporan.pdf` dan dokumentasi kontribusi.

---

## 2. Daftar Pull Request (PR) Kontribusi

Seluruh pengembangan dilakukan melalui alur cabang (*branch*) dan *fork* masing-masing anggota sebelum digabungkan ke branch `main` repositori utama melalui Pull Request:

* **PR #1 – Add UTS Machine Learning Notebook (oleh `@cheyzaamanda`):**
  * Link: https://github.com/cheyzaamanda/UTS-ML-2026/pull/1
  * Ringkasan: Penambahan notebook awal pengerjaan UTS Machine Learning.
* **PR #2 – Update kode dan hasil prediksi regresi dan klasifikasi (oleh `@zeezaayy`):**
  * Link: https://github.com/cheyzaamanda/UTS-ML-2026/pull/2
  * Ringkasan: Menambahkan implementasi kode regresi serta klasifikasi beserta keluaran prediksi (`prediksi_regresi.csv` dan `prediksi_klasifikasi.csv`).
* **PR #3 – Finalisasi laporan PDF, skrip kode, hasil clustering, dan dokumentasi kontribusi (oleh `@zeezaayy`& `@cheyzaamanda`):**
  * Link: https://github.com/cheyzaamanda/UTS-ML-2026/pull/3
  * Ringkasan: Mengunggah `laporan.pdf`, merapikan seluruh skrip di folder `kode/` beserta `README.md`, menambahkan `cluster.csv`, serta menambahkan `kontribusi.md`.
* **Commit langsung ke `main` (oleh `@cheyzaamanda`):**
  * `cluster.csv`: keluaran clustering Task 3 untuk 120.000 pelanggan.
  * `profil_segmen.csv`: ringkasan profil segmen hasil clustering.

---

## 3. Deklarasi Penggunaan AI Tools

Dalam pengerjaan tugas ini, kelompok kami menggunakan bantuan *AI tools* secara terbatas dan bertanggung jawab sesuai ketentuan akademik:

1. **AI Tools yang Digunakan:**
   * **Google Gemini** dan **ChatGPT / GitHub Copilot**
2. **Bagian yang Dibantu:**
   * **Eksplorasi & Pemahaman Konsep Statistik/ML:** Diskusi konseptual mengenai desain *expanding-window Time-Based CV*, *Split Conformal Prediction* untuk interval kuantil P90, transformasi *Centered Log-Ratio* (CLR) pada data kompositional, serta formulasi *cost-based decision threshold*.
   * **Bantuan Teknis Pemrograman & Debugging:** Membantu menyusun kerangka fungsi modular Python (seperti perhitungan *pinball loss*, $\eta^2$, dan *rolling window* kausal), memperbaiki *error* kompatibilitas versi library (`scikit-learn` / `LightGBM`), serta membantu perapihan struktur penulisan `laporan.pdf`, `README.md`, dan `kontribusi.md`.
   * **Manajemen Git & GitHub:** Panduan alur sinkronisasi *fork*, *branching*, dan pembuatan Pull Request.
3. **Cara Kelompok Memverifikasi Output:**
   * **Eksekusi dan Pengujian Mandiri:** Seluruh potongan kode yang disarankan AI tidak disalin mentah, melainkan diuji langsung pada dataset resmi (`regresi_train`, `klasifikasi_train`, `clustering`) menggunakan *random seed* tetap (`SEED = 42`).
   * **Validasi Logika & Anti-Leakage:** Setiap fitur dan transformasi diverifikasi menggunakan *assertion* dan pemeriksaan manual untuk memastikan tidak ada informasi masa depan (*data leakage*) yang masuk ke fitur latih, serta memastikan seluruh keluaran CSV (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`) bebas dari `NaN`/`Inf` dan memenuhi batasan nilai yang diminta.
   * **Pemeriksaan Konsistensi Angka:** Seluruh metrik evaluasi, tabel hasil tuning, dan grafik di dalam `laporan.pdf` dicocokkan langsung dengan keluaran eksekusi notebook/skrip final sehingga dapat direproduksi sepenuhnya oleh setiap anggota kelompok.

---

## 4. Pernyataan Persetujuan Anggota

Kami menyatakan bahwa seluruh informasi pembagian peran, tautan kontribusi GitHub, dan deklarasi penggunaan *AI tools* di atas adalah benar, telah disetujui bersama oleh seluruh anggota kelompok, dan setiap anggota mampu menjelaskan serta mereproduksi seluruh hasil pekerjaan kelompok pada saat ujian lisan (*oral exam*).

* **Disetujui oleh:**
  * [x] **Cheyza Amanda Syafira** (`@cheyzaamanda`)
  * [x] **Rafaini Zahra Syazwina** (`@zeezaayy`)
