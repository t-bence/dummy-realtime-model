# Databricks notebook source
# MAGIC %md
# MAGIC # Create model with Unity Catalog access
# MAGIC
# MAGIC Check if the deployed model can access Unity Catalog features

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

import mlflow.pyfunc

# COMMAND ----------

class UCModel(mlflow.pyfunc.PythonModel):
  def predict(self, context, input):
    df = spark.read.table("bence_toth.bubi_project.current_predictions")
    return df.take(1).asDict()["station_id"]

# COMMAND ----------

model = UCModel()

# COMMAND ----------

from mlflow.models.signature import infer_signature
import pandas as pd

X = pd.DataFrame(((0., 1.),), columns=["x", "y"])

signature = infer_signature(
  X,
  pd.DataFrame(((0.,),), columns=["prediction"])
)

# COMMAND ----------

mlflow.set_experiment(f"/Users/bence.toth@datapao.com/Dummy-Model-Tracking")

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=model,
        signature=signature,
        input_example=X,
        registered_model_name="bence_toth.testing.uc_model"
    )

# COMMAND ----------


