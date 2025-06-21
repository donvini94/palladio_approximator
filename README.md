# Performance Prediction from DSL Models

This project predicts system performance metrics (avg/min/max response time) from software architecture DSL files.

---

## 🔧 Features

- 🧾 Summary statistics prediction (avg, min, max response time)
- 🔍 Embedding methods:
  - `TF-IDF`
  - `BERT (bert-base-uncased)`
  - `LLaMA` (Code-LLaMA 7B)
- 🧠 Model types:
  - `Random Forest`
  - `Ridge` / `Lasso` Regression
  - `PyTorch Neural Network`
- 🔄 Dataset and feature caching for faster iterations
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
├── feature_extraction.py   # TF-IDF, BERT, and LLaMA embeddings
├── features/
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


### ➤ Dataset Caching (enabled by default)
```bash
# Save/load dataset from cache (default)
python train.py --model rf --embedding tfidf

# Disable dataset caching
python train.py --model rf --embedding tfidf --no_load_dataset --no_save_dataset
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
- `data/cache/dataset_splits.pkl`: Cached dataset (if enabled)
- `features/*.pkl`: Cached extracted features (if enabled)
- `features/llama_embeddings/*.npy`: Pre-computed LLaMA embeddings (if generated)
- `features/llama_embeddings/embedding_metadata.json`: Metadata for pre-computed embeddings

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




































  File Breakdown & Usage Guide

  1. Core Training & Experiment Files

  train.py - Main training script
  - Purpose: Train models and log to MLflow
  - Usage: python train.py --model rf --embedding tfidf --prediction_mode summary
  - Output: Model artifacts, MLflow metrics

  summarize_experiments.py - Experiment analysis
  - Purpose: Extract and summarize all MLflow experiments
  - Usage: python summarize_experiments.py --output results.csv --format both
  - Output: CSV summaries, markdown reports

  2. Visualization & Figure Generation

  utils/visualize.py - Main visualization module
  - Purpose: Generate all plots and figures
  - Functions:
    - create_experiment_dashboard() - Overall performance dashboard
    - prediction_error_analysis() - Prediction vs actual plots
    - visualize_embedding_space() - Embedding visualizations
    - generate_learning_curves() - Learning curve analysis

  visualize_training.py - Training visualization script
  - Purpose: Visualize training metrics from MLflow
  - Usage: python visualize_training.py
  - Output: Training loss curves, metric plots

  3. Analysis & Metrics Files

  analyze.py - Deep analysis script
  - Purpose: Comprehensive model analysis
  - Usage: python analyze.py --model_path model.pkl
  - Output: Performance analysis, error breakdowns

  utils/metrics_context.py - Advanced metrics
  - Purpose: Baseline comparisons, performance interpretation
  - Functions: Baseline metrics, normalized metrics, performance categorization

  4. Data Processing

  dataset.py - Dataset management
  - Purpose: Load/process Palladio data
  - Auto-used: By train.py and other scripts

  feature_extraction.py - Feature generation
  - Purpose: TF-IDF, BERT, LLaMA embeddings
  - Auto-used: By train.py

  How to Get Your Metrics & Figures

  Step 1: Train Models

  # Train different model configurations
  python train.py --model rf --embedding tfidf --use_mlflow
  python train.py --model svm --embedding bert --use_mlflow
  python train.py --model torch --embedding llama --use_mlflow

  Step 2: Generate Experiment Summary

  # Get comprehensive experiment results
  python summarize_experiments.py --output thesis_results --format both

  # Filter specific models
  python summarize_experiments.py --filter-model rf --output rf_results.csv

  Step 3: Generate Figures

  # Method 1: Use visualization test script
  python run_visualization_test.sh

  # Method 2: Create custom visualizations
  python -c "
  from utils.visualize import create_experiment_dashboard, prediction_error_analysis
  create_experiment_dashboard()  # Creates performance comparison plots
  "

  # Method 3: Generate training curves
  python visualize_training.py

  Step 4: Analysis & Interpretation

  # Deep analysis of best model
  python analyze.py --model_path best_model.pkl

  # Get baseline comparisons and interpretations
  python -c "
  from utils.metrics_context import get_baseline_metrics, create_metrics_interpretation
  # [analysis code]
  "

  Expected Output Locations

  - Figures: figures/ directory
    - performance/ - Model comparison plots
    - predictions/ - Prediction vs actual plots
    - embeddings/ - Embedding visualizations
    - learning_curves/ - Training progression plots
  - Experiment Data:
    - experiment_summary_[timestamp].csv - All experiments
    - experiment_summary_[timestamp].md - Markdown report
  - MLflow: mlruns/ directory (view with mlflow ui)

  Quick Start for Thesis Figures

  # 1. Train a few models
  python train.py --model rf --embedding tfidf
  python train.py --model svm --embedding bert
  python train.py --model torch --embedding llama

  # 2. Generate all figures and summaries
  python summarize_experiments.py --format both
  python run_visualization_test.sh

  # 3. Check outputs
  ls figures/  # Your plots are here
  cat experiment_summary_*.md  # Your results summary
