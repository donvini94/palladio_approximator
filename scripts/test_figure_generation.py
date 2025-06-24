#!/usr/bin/env python3
"""
Test script to generate sample SVM analysis figures with mock data.
This shows the format and style of figures that will be generated.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style for thesis
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure matplotlib for high-quality figures
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True
})

def generate_mock_svm_data():
    """Generate mock SVM experiment data for testing."""
    np.random.seed(42)
    
    # Parameters to test
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    kernel_values = ['linear', 'rbf', 'poly', 'sigmoid']
    epsilon_values = [0.001, 0.01, 0.1, 1.0]
    gamma_values = ['scale', 'auto', '0.001', '0.01', '0.1', '1.0']
    embeddings = ['tfidf', 'bert', 'llama']
    normalize_options = [True, False]
    
    data = []
    
    # Generate data for each parameter
    for param_name, param_values in [
        ('C', c_values), 
        ('kernel', kernel_values), 
        ('epsilon', epsilon_values), 
        ('gamma', gamma_values)
    ]:
        for embedding in embeddings:
            for normalize in normalize_options:
                for param_value in param_values:
                    # Generate 3 runs per combination (as per multiple runs setup)
                    for run in range(3):
                        # Simulate realistic performance with some parameter effects
                        base_mse = np.random.uniform(20, 40)
                        base_r2 = np.random.uniform(0.05, 0.25)
                        
                        # Add parameter-specific effects
                        if param_name == 'C':
                            # C=1.0 tends to be optimal
                            if param_value == 1.0:
                                base_mse *= 0.8
                                base_r2 *= 1.3
                            elif param_value in [0.01, 100.0]:
                                base_mse *= 1.2
                                base_r2 *= 0.8
                                
                        elif param_name == 'kernel':
                            # RBF kernel tends to be better
                            if param_value == 'rbf':
                                base_mse *= 0.85
                                base_r2 *= 1.2
                            elif param_value == 'sigmoid':
                                base_mse *= 1.3
                                base_r2 *= 0.7
                                
                        # Embedding effects
                        if embedding == 'bert':
                            base_mse *= 0.9
                            base_r2 *= 1.1
                        elif embedding == 'llama':
                            base_mse *= 0.85
                            base_r2 *= 1.15
                            
                        # Normalization effects
                        if normalize:
                            base_mse *= 0.95
                            base_r2 *= 1.05
                        
                        # Add noise
                        mse_noise = np.random.normal(0, base_mse * 0.1)
                        r2_noise = np.random.normal(0, base_r2 * 0.1)
                        
                        test_mse = max(0.1, base_mse + mse_noise)
                        test_r2 = max(0.001, min(0.99, base_r2 + r2_noise))
                        test_mae = test_mse * 0.7 + np.random.normal(0, 1)
                        test_rmse = np.sqrt(test_mse)
                        
                        val_mse = test_mse * np.random.uniform(0.9, 1.1)
                        val_r2 = test_r2 * np.random.uniform(0.9, 1.1)
                        val_mae = test_mae * np.random.uniform(0.9, 1.1)
                        val_rmse = np.sqrt(val_mse)
                        
                        data.append({
                            f'params.{param_name}': param_value,
                            'params.embedding': embedding,
                            'normalize_targets': normalize,
                            'test_mse': test_mse,
                            'test_mae': test_mae,
                            'test_r2': test_r2,
                            'test_rmse': test_rmse,
                            'val_mse': val_mse,
                            'val_mae': val_mae,
                            'val_r2': val_r2,
                            'val_rmse': val_rmse,
                            'run_id': f'run_{len(data)}',
                            'param_name': param_name
                        })
    
    return pd.DataFrame(data)

def create_sample_boxplot(data, param, metric, metric_name, embedding, normalize, output_dir, better):
    """Create a sample box plot."""
    param_names = {
        'C': 'Regularization Parameter C',
        'kernel': 'Kernel Type',
        'epsilon': 'SVR Epsilon Parameter',
        'gamma': 'Kernel Coefficient Gamma'
    }
    
    # Filter data for this specific combination
    filtered_data = data[
        (data['param_name'] == param) &
        (data['params.embedding'] == embedding) & 
        (data['normalize_targets'] == normalize)
    ].copy()
    
    if filtered_data.empty:
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get parameter values and sort appropriately
    param_col = f'params.{param}'
    if param in ['C', 'epsilon']:
        # Numeric parameters
        param_values = sorted(filtered_data[param_col].unique())
    else:
        # Categorical parameters
        param_orders = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', '0.001', '0.01', '0.1', '1.0']
        }
        param_values = param_orders.get(param, sorted(filtered_data[param_col].unique()))
    
    # Prepare data for box plot
    box_data = []
    labels = []
    for param_value in param_values:
        values = filtered_data[filtered_data[param_col] == param_value][metric]
        if len(values) > 0:
            box_data.append(values)
            labels.append(str(param_value))
    
    if not box_data:
        return
    
    # Create box plot with traditional rectangular boxes
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, notch=False)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(box_data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set appropriate y-axis limits based on data range
    all_values = np.concatenate(box_data)
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    y_margin = y_range * 0.1  # 10% margin
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Highlight best and worst with clearer labeling
    medians = [np.median(values) for values in box_data]
    if better == 'lower':
        best_idx = np.argmin(medians)
        worst_idx = np.argmax(medians)
        best_label = f'Best (lowest {metric_name})'
        worst_label = f'Worst (highest {metric_name})'
    else:
        best_idx = np.argmax(medians)
        worst_idx = np.argmin(medians)
        best_label = f'Best (highest {metric_name})'
        worst_label = f'Worst (lowest {metric_name})'
    
    bp['boxes'][best_idx].set_facecolor('darkgreen')
    bp['boxes'][best_idx].set_alpha(0.7)
    bp['boxes'][worst_idx].set_facecolor('darkred')
    bp['boxes'][worst_idx].set_alpha(0.7)
    
    # Customize plot
    norm_str = "with normalization" if normalize else "without normalization"
    embedding_str = embedding.upper() if embedding == 'tfidf' else embedding.capitalize()
    
    ax.set_title(f'Effect of {param_names[param]} on {metric_name}\n'
                f'{embedding_str} Embedding ({norm_str})', fontsize=14, fontweight='bold')
    ax.set_xlabel(param_names[param], fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    
    # Add sample size annotations
    for i, values in enumerate(box_data):
        n = len(values)
        ax.text(i+1, y_max + y_margin * 0.5, f'n={n}', ha='center', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', alpha=0.7, label=best_label),
        Patch(facecolor='darkred', alpha=0.7, label=worst_label)
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # Format and save
    plt.xticks(rotation=45 if len(labels) > 4 else 0)
    plt.tight_layout()
    
    # Create filename
    norm_suffix = "_norm" if normalize else "_no_norm"
    filename = f"sample_svm_{param}_{metric}_{embedding}{norm_suffix}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated sample: {filename}")

def main():
    # Create output directory
    output_dir = Path("sample_thesis_figures")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating sample SVM analysis figures...")
    
    # Generate mock data
    data = generate_mock_svm_data()
    print(f"Generated {len(data)} mock experiment runs")
    
    # Define metrics to test
    metrics = [
        ('test_mse', 'Test MSE', 'lower'),
        ('test_r2', 'Test R²', 'higher'),
    ]
    
    # Generate a subset of figures as examples
    params_to_show = ['C', 'kernel']
    embeddings_to_show = ['tfidf', 'bert']
    
    figure_count = 0
    for param in params_to_show:
        for metric, metric_name, better in metrics:
            for embedding in embeddings_to_show:
                for normalize in [True, False]:
                    create_sample_boxplot(data, param, metric, metric_name, 
                                        embedding, normalize, output_dir, better)
                    figure_count += 1
    
    print(f"\nGenerated {figure_count} sample figures in {output_dir}")
    print("\nSample figures demonstrate:")
    print("- Box plots with median, quartiles, and outliers")
    print("- Color coding for best/worst performing parameters")
    print("- Professional thesis-quality formatting")
    print("- Proper titles and axis labels")
    print("- Sample size annotations")
    print("\nTo generate real figures with your SVM experiment data:")
    print("./create_thesis_figures.sh")

if __name__ == "__main__":
    main()