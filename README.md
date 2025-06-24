# Palladio Performance Approximator

Machine learning-based performance prediction for Palladio Component Model (PCM) architecture specifications. This project predicts system performance metrics (response times) from software architecture DSL files using various ML models and embedding techniques.

## 🚀 Quick Start

### Setup
```bash
# Using Nix (recommended for development)
nix develop

# Using Docker (recommended for experiments)
scripts/run.sh build

# Using pip
pip install -r requirements.txt
```

### Basic Training
```bash
# Train Random Forest with BERT embeddings
python train.py --model rf --embedding bert --prediction_mode summary

# Train SVM with TF-IDF embeddings
python train.py --model svm --embedding tfidf --prediction_mode summary
```

## 📊 Thesis Experiments (Recommended)

The `run_thesis_experiments.sh` script provides systematic parameter sweeps for thesis analysis:

### Available Experiment Sets

```bash
# Complete SVM hyperparameter analysis
bash scripts/run_thesis_experiments.sh --experiment-set svm-analysis --baseline-embedding bert

# Random Forest optimization
bash scripts/run_thesis_experiments.sh --experiment-set rf-optimization --baseline-embedding bert

# Model architecture comparison
bash scripts/run_thesis_experiments.sh --experiment-set model-comparison

# Neural network optimization
bash scripts/run_thesis_experiments.sh --experiment-set neural-optimization --baseline-embedding bert

# Linear model analysis
bash scripts/run_thesis_experiments.sh --experiment-set linear-analysis

# Complete analysis (all experiments - long running!)
bash scripts/run_thesis_experiments.sh --experiment-set complete-analysis
```

### Individual Parameter Studies

```bash
# SVM regularization parameter impact
bash scripts/run_thesis_experiments.sh --experiment-set svm-regularization

# SVM kernel comparison
bash scripts/run_thesis_experiments.sh --experiment-set svm-kernels

# Embedding method comparison
bash scripts/run_thesis_experiments.sh --experiment-set embedding-impact

# Target normalization impact
bash scripts/run_thesis_experiments.sh --experiment-set normalization-impact
```

### Options
```bash
# Dry run to see what would be executed
bash scripts/run_thesis_experiments.sh --experiment-set svm-analysis --dry-run

# Specify number of runs per experiment
bash scripts/run_thesis_experiments.sh --experiment-set rf-optimization --n-runs 5

# Override baseline configuration
bash scripts/run_thesis_experiments.sh --experiment-set svm-analysis --baseline-model svm --baseline-embedding bert
```

## 🔧 Manual Training & Hyperparameter Search

### Core Training Script

The `train.py` script supports extensive configuration:

```bash
# Basic usage
python train.py --model MODEL --embedding EMBEDDING --prediction_mode MODE

# With hyperparameter optimization
python train.py --model svm --embedding bert --optimize_hyperparameters --n_trials 50

# With specific parameters
python train.py --model svm --embedding bert --C 10.0 --kernel rbf --gamma scale
```

### Available Models
- **`rf`** - Random Forest
- **`svm`** - Support Vector Machine
- **`ridge`** - Ridge Regression  
- **`lasso`** - Lasso Regression
- **`torch`** - Neural Network (PyTorch)

### Available Embeddings
- **`tfidf`** - TF-IDF vectorization
- **`bert`** - BERT embeddings (bert-base-uncased)
- **`llama`** - CodeLLaMA embeddings

### Hyperparameters by Model

**Random Forest:**
```bash
python train.py --model rf --n_estimators 100 --max_depth 10
```

**SVM:**
```bash
python train.py --model svm --C 1.0 --epsilon 0.1 --kernel rbf --gamma scale
```

**Neural Network:**
```bash
python train.py --model torch --epochs 200 --batch_size 128 --learning_rate 0.001 --dropout_rate 0.3
```

**Linear Models:**
```bash
python train.py --model ridge --alpha 1.0
python train.py --model lasso --alpha 0.1
```

### Hyperparameter Optimization

Enable automatic hyperparameter search:
```bash
# Bayesian optimization with Optuna
python train.py --model svm --embedding bert --optimize_hyperparameters --n_trials 100

# Specify optimization metric
python train.py --model rf --optimize_hyperparameters --optimization_metric val_r2
```

### Advanced Options

```bash
# Target normalization
python train.py --model svm --normalize_targets

# GPU usage (for neural networks and BERT)
python train.py --model torch --embedding bert --use_cuda

# MLflow experiment tracking
python train.py --model rf --use_mlflow

# Architecture comparison for neural networks
python train.py --model torch --compare_architectures --architectures_to_compare embedding_regressor,standard,residual
```

## 📈 Results Analysis

### Experiment Summarization
```bash
# Generate comprehensive results summary
python scripts/summarize_experiments.py --format both

# Filter by model or embedding
python scripts/summarize_experiments.py --filter-model rf --format csv
python scripts/summarize_experiments.py --filter-embedding bert --format markdown
```

### Figure Generation
```bash
# Generate thesis-quality figures for specific models
bash scripts/create_thesis_figures.sh --model-type svm --test-metrics
bash scripts/create_thesis_figures.sh --model-type rf --output-dir my_figures

# Generate all model figures
bash scripts/create_thesis_figures.sh --model-type all
```

### MLflow UI
```bash
# Start MLflow web interface
scripts/run.sh mlflow
# Open http://localhost:5000 in browser
```

## 🏗️ Project Structure

```
palladio_approximator/
├── scripts/                     # Execution scripts
│   ├── run_thesis_experiments.sh  # Systematic parameter sweeps
│   ├── run_parameter_sweep.sh     # Single parameter analysis
│   ├── create_thesis_figures.sh   # Figure generation
│   ├── run.sh                     # Main Docker interface
│   └── *.py                      # Utility scripts
├── models/                      # Model implementations
│   ├── rf_model.py             # Random Forest
│   ├── svm_model.py            # Support Vector Machine
│   ├── linear_model.py         # Ridge/Lasso
│   └── torch_model.py          # Neural Networks
├── utils/                       # Utilities
│   ├── visualize.py            # Plotting functions
│   ├── model_trainer.py        # Training logic
│   └── hyperparameter_optimization.py
├── data/                        # Datasets
├── train.py                     # Main training script
├── evaluate.py                  # Model evaluation
└── dataset.py                   # Data loading
```

## 🔬 Data Requirements

- **Input:** `.tpcm` files (Palladio Component Models) in `data/` or `PCMs/`
- **Labels:** Performance measurements (response times)
- **Format:** Matched pairs - `model.tpcm` requires corresponding performance data

## 📊 Output

- **Models:** Saved in `models/saved/`
- **Figures:** Generated in `thesis_figures/` or `figures/`
- **Results:** CSV/Markdown summaries via `summarize_experiments.py`
- **MLflow:** Experiment tracking in `mlruns/`

## 🎯 Thesis Workflow Recommendation

1. **Systematic Experiments:**
   ```bash
   bash scripts/run_thesis_experiments.sh --experiment-set svm-analysis --baseline-embedding bert
   bash scripts/run_thesis_experiments.sh --experiment-set model-comparison
   ```

2. **Generate Figures:**
   ```bash
   bash scripts/create_thesis_figures.sh --model-type all --test-metrics
   ```

3. **Results Analysis:**
   ```bash
   python scripts/summarize_experiments.py --format both
   scripts/run.sh mlflow  # for detailed analysis
   ```

This workflow provides systematic, reproducible experiments suitable for thesis documentation.