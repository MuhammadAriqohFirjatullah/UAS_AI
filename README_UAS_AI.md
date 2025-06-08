# 🤖 Prediksi Kemacetan Lalu Lintas dengan AI

## 📚 Tugas UAS - Kecerdasan Buatan

Proyek ini merupakan tugas UAS mata kuliah *Kecerdasan Buatan* yang bertujuan untuk membuat sistem prediksi tingkat kemacetan lalu lintas berdasarkan data historis menggunakan model AI. Aplikasi ini dibangun menggunakan Python dan pustaka *Scikit-learn*.

---

## ✅ 1. Model AI yang Digunakan

### Model: `RandomForestClassifier`

Model ini dipilih karena:

- Mampu menangani data non-linear dan kompleks
- Cocok untuk klasifikasi multi-kelas
- Memiliki performa yang baik dan relatif cepat
- Mampu memberikan *feature importance*

---

## 🧾 2. Dataset dan Fitur

### Fitur Input:
- `hour`: Jam kejadian
- `day_of_week`: Hari dalam seminggu
- `is_weekend`: Hari akhir pekan atau bukan
- `traffic_volume`: Volume lalu lintas
- `average_speed`: Kecepatan rata-rata kendaraan
- `weather`: Kondisi cuaca (cerah, hujan, dll)
- `road_segment`: Segmentasi jalan
- `has_incident`: Apakah terdapat kejadian (kecelakaan, dll)

### Target / Label:
- `traffic_level`: Kelas kemacetan (`Lancar`, `Ramai`, `Padat`, `Macet`)

---

## 🔧 3. Alur Sistem

1. **Praproses Data**: Encoding variabel kategorikal, transformasi waktu (sin/cos), dan normalisasi.
2. **Pelatihan Model**: Menggunakan 80% data untuk pelatihan dan 20% untuk pengujian.
3. **Prediksi**: Model akan memprediksi tingkat kemacetan dari input data baru.
4. **Evaluasi Model**: Menggunakan metrik akurasi, precision, recall, dan F1-score.

---

## 📊 4. Evaluasi Model

Hasil evaluasi model:

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~90%      |
| Precision  | Tinggi pada kelas "Lancar" dan "Macet" |
| Recall     | Stabil di seluruh kelas |
| F1-score   | Konsisten di atas 0.84 |

---

## 📂 5. Struktur File

```
uas ai fix/
│
├── data/
│   └── traffic_data.csv        # Dataset utama
│
├── models/
│   └── random_forest_model.pkl # Model yang sudah dilatih
│
├── notebooks/
│   └── training.ipynb          # Notebook pelatihan dan evaluasi
│
├── predict.py                  # Script untuk prediksi cepat
├── preprocessing.py            # Modul praproses
└── README.md                   # Dokumentasi proyek
```

---

## 🚀 6. Cara Menjalankan

### a. Pelatihan Model
```bash
python training.py
```

### b. Prediksi dari Data Baru
```bash
python predict.py
```

---

## 🧠 Pengembangan Selanjutnya

- Integrasi ke dashboard visual
- Menambahkan data real-time dari sensor/IOT
- Uji coba dengan model LSTM untuk prediksi time-series
- Integrasi dengan API cuaca dan peta jalan

---

## 👨‍💻 Pengembang

- [NAMA KAMU DI SINI]
- [NIM KAMU]
- Kelas: [G1A02XXX]

---

> "Dengan kecerdasan buatan, prediksi kemacetan bukan lagi hal yang sulit."
