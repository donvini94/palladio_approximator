import argparse
from dataset import load_dataset
from feature_extraction import build_tfidf_features
from models.rf_model import train_random_forest
from sklearn.metrics import mean_squared_error
import joblib


def main(data_dir):
    train_samples, val_samples, test_samples = load_dataset(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = build_tfidf_features(
        train_samples, val_samples, test_samples
    )

    rf_model = train_random_forest(X_train, y_train)

    val_pred = rf_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_pred, multioutput="raw_values")
    print("Validation MSE (avg, min, max):", val_mse)

    test_pred = rf_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred, multioutput="raw_values")
    print("Test MSE (avg, min, max):", test_mse)

    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    main(args.data_dir)
