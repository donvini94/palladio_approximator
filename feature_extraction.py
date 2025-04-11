from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_tfidf_features(train_samples, val_samples, test_samples):
    """
    Builds TF-IDF feature matrices and target arrays from dataset samples.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, vectorizer
    """
    # Extract DSL text
    train_texts = [s["tpcm_text"] for s in train_samples]
    val_texts = [s["tpcm_text"] for s in val_samples]
    test_texts = [s["tpcm_text"] for s in test_samples]

    # Create vectorizer and transform text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    print(X_train.shape)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Create target arrays (multi-output regression)
    def extract_targets(samples):
        return np.array(
            [
                [s["avg_resp_time"], s["min_resp_time"], s["max_resp_time"]]
                for s in samples
            ]
        )

    y_train = extract_targets(train_samples)
    y_val = extract_targets(val_samples)
    y_test = extract_targets(test_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer
