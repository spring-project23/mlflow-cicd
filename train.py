import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("CI-CD-RandomForest")

np.random.seed(42)

X = np.random.rand(1000, 5) * 100
y = (
    3 * X[:, 0]
    + 2 * X[:, 1]
    - X[:, 2]
    + np.random.randn(1000) * 10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=80,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

with mlflow.start_run():
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    mlflow.log_param("n_estimators", 80)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("test_r2", test_r2)

    if test_r2 < 0.95:
        raise ValueError("Model quality too low")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="CICDRandomForestModel"
    )

print("Model trained & registered")
