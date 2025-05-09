"""
Feature management module for loading and saving dataset features.
"""
import os
import joblib
import torch
import numpy as np
from utils.config import get_checkpoint_paths, get_device

from dataset import load_dataset
from feature_extraction import (
    build_tfidf_features,
    build_bert_features,
    build_llama_features,
    extract_features as feature_extraction_func  # Import the actual extraction function
)


def load_features_from_checkpoint(checkpoint_path, embedding_model_path=None):
    """Load features from a previously saved checkpoint.

    Args:
        checkpoint_path (str): Path to the feature checkpoint file
        embedding_model_path (str, optional): Path to the embedding model file

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, embedding_model, status)
            where status is True if loading was successful, False otherwise
    """
    print(f"Loading features from {checkpoint_path}...")
    embedding_model = None
    
    try:
        feature_checkpoint = joblib.load(checkpoint_path)
        X_train = feature_checkpoint["X_train"]
        y_train = feature_checkpoint["y_train"]
        X_val = feature_checkpoint["X_val"]
        y_val = feature_checkpoint["y_val"]
        X_test = feature_checkpoint["X_test"]
        y_test = feature_checkpoint["y_test"]

        # Load embedding model if available
        if embedding_model_path and os.path.exists(embedding_model_path):
            print(f"Loading embedding model from {embedding_model_path}...")
            embedding_model = joblib.load(embedding_model_path)
        else:
            print("Embedding model not found. Only features will be available.")

        print(f"Loaded features successfully: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"X_val={X_val.shape}, X_test={X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model, True

    except Exception as e:
        print(f"Error loading features: {e}")
        print("Falling back to generating features from scratch")
        return None, None, None, None, None, None, None, False


def save_features(checkpoint_path, embedding_model_path, 
                 X_train, y_train, X_val, y_val, X_test, y_test, embedding_model=None):
    """Save extracted features and embedding model to disk.

    Args:
        checkpoint_path (str): Path to save the feature checkpoint
        embedding_model_path (str): Path to save the embedding model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        embedding_model: The embedding model used for feature extraction
    """
    print(f"Saving extracted features to {checkpoint_path}...")
    feature_checkpoint = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(feature_checkpoint, checkpoint_path)
    print(f"Features saved successfully")

    # Also save the embedding model for completeness
    if embedding_model is not None:
        print(f"Saving embedding model to {embedding_model_path}...")
        joblib.dump(embedding_model, embedding_model_path)


def extract_features(args, device):
    """Extract features based on the specified embedding type.

    Args:
        args: Command line arguments containing embedding options
        device (str): 'cuda' or 'cpu' for processing

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, embedding_model)
    """
    # Use the imported function from feature_extraction.py
    # This ensures we're using the latest implementation that supports pre-computed embeddings
    return feature_extraction_func(args, device)


def get_features(args):
    """Load or extract features based on command-line arguments.

    Args:
        args: Command line arguments

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, embedding_model)
    """
    checkpoint_path, embedding_model_path = get_checkpoint_paths(args)
    
    # First check if we should load features from disk
    if args.load_features and os.path.exists(checkpoint_path):
        X_train, y_train, X_val, y_val, X_test, y_test, embedding_model, success = load_features_from_checkpoint(
            checkpoint_path, embedding_model_path
        )
        
        if success:
            return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model
    elif args.load_features:
        print(f"Feature checkpoint {checkpoint_path} not found. Generating features.")
    
    # Extract features if we couldn't load them
    device = get_device(args)
    X_train, y_train, X_val, y_val, X_test, y_test, embedding_model = extract_features(args, device)
    
    # Save embeddings as checkpoint if requested
    if args.save_features:
        save_features(
            checkpoint_path, 
            embedding_model_path,
            X_train, y_train, X_val, y_val, X_test, y_test, 
            embedding_model
        )
        
    return X_train, y_train, X_val, y_val, X_test, y_test, embedding_model