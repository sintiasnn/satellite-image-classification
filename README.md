# Satellite Image Classification API

API untuk klasifikasi citra satelit menggunakan TensorFlow CNN model.

**Kelas:** cloudy, desert, green_area, water

## Setup

```bash
pip install -r requirements.txt
```

## Menjalankan API

```bash
uvicorn src.main:app --reload
```

Akses dokumentasi interaktif di http://localhost:8000/docs

## Endpoints

| Method | Path              | Description                      |
|--------|-------------------|----------------------------------|
| GET    | `/`               | Root info                        |
| GET    | `/api/v1/health`  | Health check                     |
| POST   | `/api/v1/predict` | Klasifikasi gambar satelit       |

### Contoh Predict

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@gambar.jpg"
```

Response:
```json
{
  "predicted_class": "green_area",
  "confidence_scores": {
    "cloudy": 0.02,
    "desert": 0.01,
    "green_area": 0.95,
    "water": 0.02
  }
}
```

## Model

Trained CNN dengan arsitektur 3 Conv2D layers + Dense layers.
Letakkan file model (`model-example.h5`) di root project atau atur path via environment variable `MODEL_PATH`.

## Struktur Project

```
src/
├── main.py              # Entry point FastAPI
├── api/routes.py        # Endpoints
├── core/config.py       # Konfigurasi
├── models/classifier.py # Load model & inference
├── schemas/prediction.py# Request/response schemas
└── utils/image_utils.py # Preprocessing gambar
```
