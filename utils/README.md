# Thesis Visualization Utilities

This directory contains utilities for creating visualizations and analyzing ML models for the master thesis on Palladio performance approximation.

## Overview of Visualization Tools

### `visualize.py`

The main visualization module that provides functions for:

- **Performance Comparison**: Compare different model configurations with bar charts and heatmaps
- **Prediction Error Analysis**: Create scatter plots, residual plots, and error distribution histograms
- **Learning Curves**: Show how model performance changes with training data size
- **Embedding Space Visualization**: Project high-dimensional embeddings to 2D/3D using PCA and t-SNE

### `attention.py`

Specialized module for analyzing attention patterns in embedding models:

- **Attention Visualization**: Create heatmaps of attention weights for different model layers
- **Token Importance Analysis**: Identify which tokens receive the most attention
- **Model Representation Analysis**: Visualize how models represent different input data
- **Keyword Analysis**: Extract and visualize attention patterns for specific keywords

## Usage

### Using the Analysis Script

The `analyze.py` script in the root directory provides a high-level interface:

```bash
# Generate all visualizations
python analyze.py --all

# Generate specific visualizations
python analyze.py --mlflow  # Analyze MLflow experiments
python analyze.py --performance  # Run performance analysis
python analyze.py --embeddings  # Analyze embedding spaces
python analyze.py --attention --code_dir=PCMs  # Analyze attention patterns
python analyze.py --learning_curves  # Generate learning curves
```

### Testing Visualizations

Use the `test_visualization.py` script to test visualization capabilities:

```bash
# Test all visualization features
python test_visualization.py --all

# Test specific features
python test_visualization.py --attention
python test_visualization.py --embedding
python test_visualization.py --prediction
```

## Visualization Types

### Performance Visualizations

- Model comparison bar charts and heatmaps
- Error distribution histograms
- Scatter plots of predicted vs. actual values
- Residual plots
- Learning curves for different metrics

### Embedding Visualizations

- 2D/3D projections of embedding spaces
- Token importance plots
- Attention heatmaps for different model layers
- Keyword attention patterns

### MLflow Integration

The visualization tools integrate with MLflow to analyze and visualize experiment results, making it easy to compare different model configurations.