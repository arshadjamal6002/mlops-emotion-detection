import mlflow
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/arshadjamal6002/mlops-emotion-detection.mlflow')
dagshub.init(repo_owner='arshadjamal6002', repo_name='mlops-emotion-detection', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1) 