from sklearn.linear_model import Ridge, Lasso


def train_linear_model(X_train, y_train, model_type="ridge", alpha=1.0):
    """
    Trains a Ridge or Lasso regression model.

    Parameters:
        X_train: Feature matrix
        y_train: Target array (single target: avg response time)
        model_type (str): 'ridge' or 'lasso'
        alpha (float): Regularization strength

    Returns:
        Trained Ridge or Lasso model
    """
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    else:
        raise ValueError("Invalid model_type. Use 'ridge' or 'lasso'.")

    model.fit(X_train, y_train)
    return model
