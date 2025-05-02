# Performance Prediction from DSL Models

This project predicts system performance metrics (avg/min/max response time) from software architecture DSL files.

---

## 🔧 Features

- 🧾 Summary statistics prediction (avg, min, max response time)
- 🔍 Embedding methods:
  - `TF-IDF`
  - `BERT (bert-base-uncased)`
- 🧠 Model types:
  - `Random Forest`
  - `Ridge` / `Lasso` Regression
  - `PyTorch Neural Network`
- 🔁 Optional MLflow experiment tracking
- 📦 Cleanly modular & easily extensible

---

## 📁 Project Structure
```
palladio_approximator/
├── data/
│   ├── dsl_models/         # .tpcm files (DSL)
│   └── measurements/       # .csv files (Response Time [s])
├── dataset.py              # Loads and preprocesses DSL + CSV
├── feature_extraction.py   # TF-IDF + BERT embeddings
├── models/
│   ├── rf_model.py         # Random forest trainer
│   ├── linear_model.py     # Ridge / Lasso trainer
│   └── torch_model.py      # PyTorch neural network model
├── evaluate.py             # Reusable model evaluation
├── train.py                # Unified training script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Use

### 1. Prepare your data
- Place `.tpcm` files in `data/dsl_models/`
- Place matching `.csv` files in `data/measurements/`
- Files must have the same base name, e.g., `foo.tpcm` + `foo.csv`
- Alternatively, use `collect_files.py` to gather data from PCMs directory

### 2. Install dependencies

#### Using Nix (for development)
```bash
nix develop
```

#### Using Docker (for experiments)
```bash
./run.sh build   # Build the Docker image
./run.sh bash    # Start a bash shell in the container
```

#### Using pip
```bash
pip install -r requirements.txt
```

---

## 🧪 Example Commands

### ➤ Train Random Forest with TF-IDF (baseline)
```bash
python train.py --model rf --embedding tfidf --prediction_mode summary
```

### ➤ Train Ridge Regression with TF-IDF
```bash
python train.py --model ridge --embedding tfidf --alpha 0.5
```

### ➤ Train with BERT embeddings on GPU
```bash
python train.py --model rf --embedding bert --use_cuda
```

### ➤ Enable MLflow tracking
```bash
python train.py --model rf --embedding bert --use_cuda --use_mlflow
```

### ➤ Using Docker
```bash
./run.sh train rf bert summary
```

---

## 📊 Output Files
- Trained models: `rf_model.pkl`, `ridge_model.pkl`, etc.
- Embedding models: `tfidf_embedding.pkl`, or BERT model/tokenizer objects
- `data/all_samples.csv`: Preprocessed dataset

---

## 🔬 Evaluation Script

You can import `evaluate.py` in any script:
```python
from evaluate import evaluate_model
results = evaluate_model(model, X_val, y_val, split_name="val")
```
This prints and returns MSE & MAE for each target (avg, min, max).

---

## 📌 Notes on Reproducibility
- All model hyperparameters are logged.
- `--use_mlflow` stores experiments with:
  - Model type
  - Embedding method
  - Prediction mode
  - Key metrics
  - Model files
- This enables detailed experiment comparisons in the MLflow UI.

---

## ⏭ Coming Soon
- `--prediction_mode timeseries` for predicting values at individual timesteps
- Cross-validation scripts and more model types (e.g. MLPs)

---

## 📫 Questions?
Open an issue or reach out!