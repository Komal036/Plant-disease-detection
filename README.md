# 🌿 Plant Disease Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://erjmtgw8ucpazavhkg95wf.streamlit.app)

> 🚀 **[Live Demo → Try it now!](https://erjmtgw8ucpazavhkg95wf.streamlit.app)**

An end-to-end deep learning system for detecting **plant diseases from leaf images**
using EfficientNetV2S Transfer Learning, TensorFlow 2.x, and a Streamlit web interface.

---

## 📌 Project Overview

Farmers lose 20–40% of global crop yield to plant diseases every year.
This system enables real-time disease diagnosis from smartphone photos,
achieving ~94% validation accuracy across 15 disease classes.

The system predicts:
- ✅ The **plant type**
- ✅ The **disease** affecting the plant
- ✅ A **confidence score** for the prediction

---

## 📂 Dataset

| Property      | Detail                                        |
|---------------|-----------------------------------------------|
| Name          | PlantVillage Dataset                          |
| Classes       | 15 plant disease categories                   |
| Total Images  | ~20,000 (subset)                              |
| Source        | https://www.kaggle.com/datasets/emmarex/plantdisease |

**Supported disease classes:**
- Pepper Bell Bacterial Spot / Healthy
- Potato Early Blight / Late Blight / Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold,
  Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus,
  Yellow Leaf Curl Virus, Healthy

---

## 🧠 Model Architecture

| Component      | Detail                           |
|----------------|----------------------------------|
| Base Model     | EfficientNetV2S (ImageNet)       |
| Input Size     | 224 × 224 × 3                    |
| Head           | GAP → BN → Dropout → Dense(256) |
| Output         | Softmax (15 classes)             |
| Training       | 2-phase fine-tuning              |
| Framework      | TensorFlow 2.15 / Keras          |

---

## 📊 Results

| Metric              | Value   |
|---------------------|---------|
| Train Accuracy      | ~94.2%  |
| Validation Accuracy | ~92.1%  |
| Macro F1 Score      | 0.918   |
| Inference Time      | ~35 ms  |
| Model Size (FP16)   | ~25 MB  |

---

## 🚀 Installation
```bash
# 1. Clone the repository
git clone https://github.com/Komal036/Plant-disease-detection
cd Plant-disease-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env if needed

# 5. Download dataset and place it at:
#    data/raw/PlantVillage/
```

---

## ⚡ How to Run

### Train the model
```bash
python main.py --mode train --config configs/config.yaml
```

### Evaluate on test set
```bash
python main.py --mode evaluate
```

### Export to TFLite (for mobile/edge deployment)
```bash
python main.py --mode export
```

### Launch Streamlit web app
```bash
streamlit run app/streamlit_app.py
```

### Run unit tests
```bash
pytest tests/ -v --tb=short
```

---

## 📁 Project Structure
```
plant-disease-detection/
├── data/
│   ├── raw/PlantVillage/      # Downloaded dataset (not in Git)
│   └── processed/             # Pre-processed images
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── models/
│   ├── saved/                 # Trained model (not in Git)
│   └── metadata.json
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── configs/
│   └── config.yaml
├── results/                   # Graphs and reports
├── .env.example
├── requirements.txt
├── main.py
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Add Grad-CAM visualisation for model explainability
- [ ] Export to TFLite for Android/iOS app
- [ ] Add disease treatment recommendations using RAG + LLM
- [ ] Train on real-world (non-lab) field images
- [ ] REST API with FastAPI + Docker

---

## 👩‍💻 Author

**Komal Kumari** · [GitHub](https://github.com/Komal036) 

---

⭐ If you found this project useful, consider giving it a star!
