from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train, **kwargs):
    """
    Trains a RandomForestRegressor on the provided data.

    Parameters:
        X_train: Feature matrix
        y_train: Target array (single target: avg response time)
        kwargs: Additional keyword args for model (e.g., n_estimators)

    Returns:
        model: Trained RandomForestRegressor
    """
    model = RandomForestRegressor(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model
