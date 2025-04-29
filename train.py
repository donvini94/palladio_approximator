import argparse
from dataset import load_dataset
from dataset import build_time_series_dataset
from feature_extraction import (
    build_tfidf_features,
    build_bert_features,
    build_longformer_features,
    build_chunk_aware_features,
)
from timeseries_features import build_timeseries_tfidf, build_timeseries_bert
from models.rf_model import train_random_forest
from models.linear_model import train_linear_model
from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error
import joblib
import torch
import mlflow


def main(args):
    if args.prediction_mode == "summary":
        train_samples, val_samples, test_samples = load_dataset(args.data_dir)

        # Embedding selection
        if args.embedding == "tfidf":
            X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = (
                build_tfidf_features(train_samples, val_samples, test_samples)
            )
            embedding_model = vectorizer
        elif args.embedding == "bert":

            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
                build_bert_features(
                    train_samples,
                    val_samples,
                    test_samples,
                    device=device,
                    chunking_strategy="hierarchical",  # or "average"
                )
            )
            embedding_model = (tokenizer, model)
        elif args.embedding == "chunk":

            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            # For DSL-aware processing
            X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
                build_chunk_aware_features(
                    train_samples, val_samples, test_samples, device=device
                )
            )
        elif args.embedding == "longformer":

            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, model = (
                build_longformer_features(
                    train_samples, val_samples, test_samples, device=device
                )
            )
            embedding_model = (tokenizer, model)
        else:
            raise ValueError("Unsupported embedding type. Choose from 'tfidf', 'bert'.")

    elif args.prediction_mode == "timeseries":
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, *_ = (
            build_time_series_dataset(args.data_dir)
        )

        if args.embedding == "tfidf":
            X_train, tfidf_vectorizer = build_timeseries_tfidf(X_train_raw)
            X_val, _ = build_timeseries_tfidf(X_val_raw, tfidf_vectorizer)
            X_test, _ = build_timeseries_tfidf(X_test_raw, tfidf_vectorizer)
            embedding_model = tfidf_vectorizer
        elif args.embedding == "bert":
            device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
            X_train, tokenizer, model = build_timeseries_bert(
                X_train_raw, device=device
            )
            X_val, _, _ = build_timeseries_bert(
                X_val_raw, tokenizer=tokenizer, model=model, device=device
            )
            X_test, _, _ = build_timeseries_bert(
                X_test_raw, tokenizer=tokenizer, model=model, device=device
            )
            embedding_model = (tokenizer, model)
        else:
            raise ValueError("Unsupported embedding type. Choose from 'tfidf', 'bert'.")
    else:
        raise ValueError(
            "Unsupported prediction mode. Choose from 'summary', 'timeseries'."
        )

    if args.use_mlflow:
        mlflow.set_experiment("dsl-performance-prediction")
        mlflow.start_run()
        mlflow.log_params(
            {
                "model_type": args.model,
                "embedding": args.embedding,
                "prediction_mode": args.prediction_mode,
                "n_estimators": args.n_estimators,
                "alpha": args.alpha,
            }
        )

    if args.model == "rf":
        model = train_random_forest(X_train, y_train, n_estimators=args.n_estimators)
    elif args.model in ("ridge", "lasso"):
        model = train_linear_model(
            X_train, y_train, model_type=args.model, alpha=args.alpha
        )
    else:
        raise ValueError("Unsupported model type. Choose 'rf', 'ridge', or 'lasso'.")

    val_results = evaluate_model(model, X_val, y_val, split_name="val")
    test_results = evaluate_model(model, X_test, y_test, split_name="test")

    if args.use_mlflow:
        mlflow.log_metrics(val_results)
        mlflow.log_metrics(test_results)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

    joblib.dump(model, f"{args.model}_{args.prediction_mode}_model.pkl")
    joblib.dump(
        embedding_model, f"{args.embedding}_{args.prediction_mode}_embedding.pkl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--model", type=str, choices=["rf", "ridge", "lasso"], default="rf"
    )
    parser.add_argument(
        "--embedding", type=str, choices=["tfidf", "bert", "longformer"], default="bert"
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["summary", "timeseries"],
        default="summary",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--no_cuda", dest="use_cuda", action="store_false", help="Disable CUDA"
    )
    parser.add_argument(
        "--no_mlflow", dest="use_mlflow", action="store_false", help="Disable MLflow"
    )
    parser.set_defaults(use_cuda=True, use_mlflow=True)
    args = parser.parse_args()
    main(args)
