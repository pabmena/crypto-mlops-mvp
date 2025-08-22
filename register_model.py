import mlflow
import mlflow.pyfunc
import pickle

class VolatilityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open("volatility_lstm.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://mlflow:5000")  # o localhost si corrés fuera del contenedor
    mlflow.set_experiment("volatility_experiment")

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="volatility_model",
            python_model=VolatilityModel(),
        )
        print("✅ Modelo registrado en MLflow")
