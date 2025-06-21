from sklearn.svm import SVR


def train_svm(X_train, y_train, **kwargs):
    """
    Trains a Support Vector Machine regressor on the provided data.

    Parameters:
        X_train: Feature matrix
        y_train: Target array with avg response time
        kwargs: Additional keyword args for model (e.g., C, epsilon, kernel)

    Returns:
        model: Trained SVR model
    """
    # Use SVR directly for single target (avg response time)
    model = SVR(**kwargs)
    model.fit(X_train, y_train)
    return model