# PPE Detection API (YOLOv8n + ONNX + FastAPI + MySQL)

API deteksi objek untuk 3 kelas: `Safety Helmet`, `Safety Vest`, `Safety Boot`, plus penyimpanan riwayat checklist ke MySQL.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/ppe_db"  # opsional, kalau tidak diset endpoint /checklist/* nonaktif
uvicorn main:app --host 0.0.0.0 --port 8000
```

Coba:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@contoh.jpg"
```

## Deploy ke Railway

1. Push folder `deploy/` ini ke sebuah repo GitHub.
2. Di Railway: **New Project -> Deploy from GitHub repo** -> pilih repo ini.
3. **Tambah database:** di project yang sama, klik **New -> Database -> Add MySQL**.
4. Buka service backend (bukan service MySQL-nya) -> tab **Variables** -> **New Variable**:
   - Name: `DATABASE_URL`
   - Value: klik **Add Reference** -> pilih service MySQL -> pilih `MYSQL_URL`
5. Railway otomatis build ulang & tabel `checklist_entries` dibuat otomatis saat backend start (lihat `@app.on_event("startup")` di `main.py`).
6. Catat URL publik backend (misal `https://xxxx.up.railway.app`) — dipakai di frontend Vercel sebagai `VITE_API_URL`.

## Endpoint

- `GET /` dan `GET /health` — cek server hidup.
- `POST /predict` — kirim `file` (multipart/form-data) berisi gambar, response JSON daftar deteksi (`class_name`, `confidence`, `box_xyxy`).
- `POST /checklist` — simpan hasil pengecekan ke MySQL. Body JSON:
  ```json
  {
    "technician": "Rayhan",
    "location": "GI Cilegon",
    "helmet": true, "vest": true, "shoes": false,
    "conf_helmet": 96.2, "conf_vest": 91.0, "conf_shoes": 0
  }
  ```
- `GET /checklist/history?limit=200` — daftar riwayat pengecekan (terbaru dulu).
- `GET /checklist/stats` — ringkasan statistik (total, approved/rejected, rata-rata confidence, data mingguan, tingkat pemakaian per APD, tren 14 hari).

## Catatan Deployment

- Dockerfile sudah menambahkan `libgl1` & `libglib2.0-0` supaya `opencv-python-headless` tidak error di Railway.
- CORS aktif (`allow_origins=["*"]`) supaya bisa dipanggil dari domain Vercel manapun. Kalau mau dibatasi, ganti ke domain Vercel spesifik di `main.py`.
- Kalau `DATABASE_URL` belum diset, endpoint `/checklist/*` akan mengembalikan `503` dengan pesan jelas, tapi `/predict` tetap berfungsi normal.
