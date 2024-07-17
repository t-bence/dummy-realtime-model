# Databricks notebook source
# MAGIC %md
# MAGIC Let's create a model and register it, so we can serve it from Databricks model serving

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

import mlflow.pyfunc

# COMMAND ----------

class Model(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return 5
    

# COMMAND ----------

model = Model()

# COMMAND ----------

# MAGIC %md
# MAGIC Generate some dummy signature, as it is needed to register the model

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
        registered_model_name="bence_toth.testing.dummy-model"
    )

# COMMAND ----------


