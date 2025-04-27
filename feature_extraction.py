from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


def extract_targets(df):
    return df[["avg_resp_time", "min_resp_time", "max_resp_time"]].to_numpy()


def build_tfidf_features(train_samples, val_samples, test_samples):
    """
    Builds TF-IDF feature matrices and target arrays from dataset samples.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, vectorizer
    """
    # Extract DSL text
    train_texts = train_samples["tpcm_text"].tolist()
    val_texts = val_samples["tpcm_text"].tolist()
    test_texts = test_samples["tpcm_text"].tolist()

    # Create vectorizer and transform text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    y_train = extract_targets(train_samples)
    y_val = extract_targets(val_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer


def build_bert_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="microsoft/codebert-base",
    device="cpu",
    batch_size=8,
):
    """
    Builds CodeBERT embeddings and target arrays from dataset samples.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(df):
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # safe cutoff for BERT-like models
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(cls_embeddings)

        return np.vstack(embeddings)

    X_train = encode(train_samples)
    y_train = extract_targets(train_samples)

    X_val = encode(val_samples)
    y_val = extract_targets(val_samples)

    X_test = encode(test_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model


def build_longformer_features(
    train_samples,
    val_samples,
    test_samples,
    model_name="allenai/longformer-base-4096",
    device="cpu",
    batch_size=2,  # bigger inputs = smaller batch size
):
    """
    Builds Longformer embeddings and target arrays from dataset samples.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def encode(df):
        texts = df["tpcm_text"].tolist()
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=4096,  # Longformer can handle up to 4096
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(cls_embeddings)

        return np.vstack(embeddings)

    X_train = encode(train_samples)
    y_train = extract_targets(train_samples)

    X_val = encode(val_samples)
    y_val = extract_targets(val_samples)

    X_test = encode(test_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model
