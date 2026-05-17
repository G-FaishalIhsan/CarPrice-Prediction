# AutoPrice Pro — Panduan Deployment

## Struktur Proyek
```
autoprice/
├── backend/
│   ├── main.py                              ← FastAPI server
│   ├── car_price_linear_regression.joblib  ← Model tersimpan
│   ├── feature_config.json                 ← Konfigurasi fitur & metrik
│   └── requirements.txt
└── frontend/
    └── index.html                          ← Frontend (buka langsung di browser)
```

## Cara Menjalankan Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
python main.py
# atau: uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Cara Menjalankan Frontend

**Opsi A — Langsung buka di browser:**
Buka `frontend/index.html` langsung di browser. Frontend sudah memiliki Demo Mode otomatis jika backend belum berjalan.

**Opsi B — Serve lokal:**
```bash
cd frontend
python -m http.server 3000
# buka: http://localhost:3000
```

## Deployment ke Cloud

### Railway / Render (Backend)
1. Push folder `backend/` ke GitHub repo
2. Tambahkan `Procfile`: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Deploy ke Railway atau Render (free tier tersedia)

### Vercel / Netlify (Frontend)
1. Upload folder `frontend/` 
2. Ubah `API_BASE` di `index.html` ke URL Railway/Render Anda

### Update `API_BASE` di frontend/index.html:
```javascript
const API_BASE = "https://your-app.railway.app";  // ganti dengan URL deployment Anda
```

## API Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Health check |
| GET | `/model-info` | Informasi model & metrik |
| POST | `/predict` | Prediksi harga mobil |

### Contoh Request POST /predict:
```json
{
  "engine_size": 2.3,
  "horsepower": 150,
  "wheelbase": 107,
  "width": 70,
  "length": 185,
  "curb_weight": 3.1,
  "fuel_capacity": 16,
  "fuel_efficiency": 27,
  "vehicle_type": "Passenger"
}
```