# 🌿 Plant Disease Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://erjmtgw8ucpazavhkg95wf.streamlit.app)

> 🚀 **[Live Demo → Try it now!](https://erjmtgw8ucpazavhkg95wf.streamlit.app)**

An end-to-end deep learning system for detecting **plant diseases from leaf images**
using EfficientNetV2S transfer learning, TensorFlow 2.x, and a Streamlit web interface.

---

## 📌 Project Overview

Farmers lose 20–40% of global crop yield to plant diseases every year.
This system enables real-time disease diagnosis from smartphone photos,
predicting the plant disease class and providing treatment recommendations.

The system predicts:
- ✅ The **plant type** and **disease** affecting the plant
- ✅ A **confidence score** for the prediction
- ✅ A **recommended treatment** for the detected disease

---

## 📂 Dataset

| Property      | Detail                                        |
|---------------|-----------------------------------------------|
| Name          | PlantVillage Dataset                          |
| Classes       | 15 plant disease categories                   |
| Total Images  | ~20,000 (subset)                              |
| Source        | https://www.kaggle.com/datasets/emmarex/plantdisease |

**Supported disease classes:**
- Pepper Bell: Bacterial Spot, Healthy
- Potato: Early Blight, Late Blight, Healthy
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold,
  Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus,
  Yellow Leaf Curl Virus, Healthy

---

## 🧠 Model Architecture

| Component      | Detail                                          |
|----------------|-------------------------------------------------|
| Base Model     | EfficientNetV2S (pretrained on ImageNet)        |
| Input Size     | 224 × 224 × 3                                   |
| Head           | GAP → BatchNorm → Dropout(0.3) → Dense(256)     |
| Output         | Softmax (15 classes)                            |
| Training       | 2-phase: head-only (10 epochs) → fine-tune (20) |
| Framework      | TensorFlow 2.16 / Keras                         |
| Preprocessing  | `efficientnet_v2.preprocess_input` (not /255)   |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Komal036/Plant-disease-detection
cd Plant-disease-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies (local CPU)
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

## ☁️ Training on Google Colab

For GPU training, use the notebook in `notebooks/Plant_Disease_Training.ipynb`.

**Quick setup:**
1. Upload `kaggle.json` to your Google Drive
2. Open the notebook in Colab (Runtime → Change runtime type → GPU T4)
3. Run Cell 2 and paste the keep-alive JS into your browser console
4. Run all cells from top to bottom

**Resuming after a disconnect:**
1. Re-run cells 1–6 (setup)
2. In Cell 7, set `RESUME = True` and set `INITIAL_EPOCH_HEAD` /
   `INITIAL_EPOCH_FINETUNE` to the last completed epochs
3. Run from Cell 7 onwards — the best checkpoint is loaded automatically

---

## 📁 Project Structure

```
plant-disease-detection/
├── app/
│   └── streamlit_app.py       # Streamlit web interface
├── configs/
│   └── config.yaml            # All hyperparameters (single source of truth)
├── data/
│   ├── raw/PlantVillage/      # Downloaded dataset (not in Git)
│   └── processed/
├── models/
│   ├── saved/                 # Trained model weights (not in Git)
│   └── metadata.json          # Class names + model config (in Git)
├── notebooks/
│   └── Plant_Disease_Training.ipynb
├── results/                   # Evaluation graphs and reports
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt           # Local / CPU
├── requirements-colab.txt     # Colab / GPU (no tensorflow)
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Grad-CAM visualisation for model explainability
- [ ] TFLite export for Android/iOS app
- [ ] Disease treatment recommendations using RAG + LLM
- [ ] Train on real-world (non-lab) field images
- [ ] REST API with FastAPI + Docker

---

## 👩‍💻 Author

**Komal Kumari** · [GitHub](https://github.com/Komal036)

---

⭐ If you found this project useful, consider giving it a star!
