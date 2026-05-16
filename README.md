# Satellite Image Classification

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Notebook](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab)

Satellite image classification using CNN (TensorFlow/Keras) served via FastAPI REST API.

**Classes:** cloudy, desert, green_area, water

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Notebook Flow](#notebook-flow)
- [Results](#results)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)
- [Author](#author)

---

## Overview

This project classifies satellite images into 4 categories using a CNN model built from scratch. The trained model is served through a FastAPI REST API for real-time inference.

**Pipeline stages:**
1. **Data Preparation** — Load and split satellite images (80/20)
2. **Data Augmentation** — Rotation, horizontal flip, shear
3. **Modeling** — 3-layer Conv2D CNN with Dropout
4. **Training** — Early stopping at >85% accuracy
5. **Serving** — FastAPI REST API with Swagger docs

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras (CNN) |
| API Framework | FastAPI |
| Image Processing | Pillow, NumPy |
| Visualization | Matplotlib |
| Environment | Google Colab (GPU) |

## Project Structure

```
satellite-image-classification/
├── deployment_image_classiffication_1.ipynb   # Training notebook
├── deployment_image_classiffication_1.py      # Notebook export
├── src/
│   ├── main.py                # FastAPI entry point
│   ├── api/routes.py          # API endpoints
│   ├── core/config.py         # Configuration
│   ├── models/classifier.py   # Model loading & inference
│   ├── schemas/prediction.py  # Request/response schemas
│   └── utils/image_utils.py   # Image preprocessing
├── models/
├── .gitignore
├── requirements.txt
└── README.md
```

> Model artifact (`model-example.h5`) is excluded from the repo. Run the notebook in Colab to generate it.

## Dataset

- **Source**: Satellite image dataset (4 classes)
- **Samples**: 4,505 training / 1,126 validation
- **Resolution**: 150×150 (RGB)
- **Split**: Training 80% / Validation 20%
- **Classes**:

| # | Class | Description |
|---|---|---|
| 1 | Cloudy | Cloud-covered satellite imagery |
| 2 | Desert | Arid / desert terrain |
| 3 | Green Area | Vegetation / forest |
| 4 | Water | Oceans, lakes, rivers |

## Notebook Flow

| # | Section |
|---|---|
| 1 | Project Description (dataset, problem statement, ML solution) |
| 2 | Kaggle API Setup (install, credentials, download dataset) |
| 3 | Data Preparation (extract, directories, augmentation, generators) |
| 4 | Model Definition (CNN architecture, compile with Adam) |
| 5 | Model Training & Evaluation (EarlyStopping, train, plot accuracy & loss) |
| 6 | Model Deployment & Utility (prediction example, save model) |

## Results

| Metric | Value |
|---|---|
| Train Accuracy | ~88% |
| Validation Accuracy | ~87–97% |
| Epochs | 12 (early stopped) |

## Getting Started

### 1. Train Model (Google Colab)

Open `deployment_image_classiffication_1.ipynb` in Google Colab with GPU runtime, then download the generated `model-example.h5`.

### 2. Setup API Locally

```bash
git clone https://github.com/sintiasnn/satellite-image-classification.git
cd satellite-image-classification
```

**Create virtual environment (wajib):**

<details>
<summary>Using venv (Python built-in)</summary>

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```
</details>

<details>
<summary>Using conda</summary>

```bash
conda create -n satelit python=3.11 -y
conda activate satelit
```
</details>

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Place `model-example.h5` in the project root directory.

### 3. Run API

```bash
uvicorn src.main:app --reload
```

Access interactive docs at http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Classify satellite image |

### Predict Example

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@gambar.jpg"
```

Response:
```json
{
  "predicted_class": "water",
  "confidence_scores": {
    "cloudy": 3.33e-15,
    "desert": 7.45e-24,
    "green_area": 0.001,
    "water": 0.998
  }
}
```

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)
