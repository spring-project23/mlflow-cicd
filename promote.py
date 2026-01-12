from mlflow.tracking import MlflowClient
import os

client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
MODEL_NAME = "CICDRandomForestModel"

versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
if not versions:
    raise Exception("No model versions found")

version = versions[0].version
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model v{version} promoted to Production")
