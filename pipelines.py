import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from google.cloud import storage, bigquery
from google.cloud import aiplatform
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from datetime import datetime

import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from google.cloud import storage, bigquery
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.simplefilter("ignore", FutureWarning)

# Componente para entrenar el modelo
@component
def train_model_op(base_image: str = 'python:3.9') -> None:
    # Crear datos de prueba y entrenar el modelo
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    model = LinearRegression()
    model.fit(X, y)

    # Guardar el modelo
    model_filename = '/tmp/trained_model.pkl'
    joblib.dump(model, model_filename)

    # Subir el modelo a Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket('vertex-ai-despliegue-en-gcp')  # Reemplaza con tu bucket
    blob = bucket.blob('models/trained_model.pkl')
    blob.upload_from_filename(model_filename)
    print("Modelo entrenado y subido a Cloud Storage.")

# Componente para validar las tablas en BigQuery
@component
def validate_bigquery_tables_op() -> None:
    client = bigquery.Client()
    dataset_id = 'project-mlops9-rbarzola.predicciones_dataset'  # Reemplazado con tu dataset
    tables = ['predicciones_dataset', 'resultados_predicciones']  # Nombres de las tablas necesarias

    missing_tables = []
    for table in tables:
        try:
            client.get_table(f'{dataset_id}.{table}')
        except Exception:
            missing_tables.append(table)

    if missing_tables:
        raise ValueError(f"Las siguientes tablas faltan: {', '.join(missing_tables)}")
    else:
        print("Todas las tablas están disponibles.")

# Componente para imputar los datos faltantes
@component
def impute_data_op() -> None:
    data = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': [None, 2, 3, 4, 5]
    })

    # Imputación de datos faltantes
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(data)
    print("Datos imputados.")

# Componente para realizar la predicción y almacenarla en BigQuery
@component
def prediction_op() -> None:
    # Cargar el modelo desde Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket('vertex-ai-despliegue-en-gcp')
    blob = bucket.blob('models/trained_model.pkl')
    
    # Descargar el archivo binario del modelo
    model = joblib.load(blob.download_as_bytes())

    # Simular datos para predicción
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

    # Realizar la predicción
    predictions = model.predict(data)

    # Almacenar en BigQuery
    bq_client = bigquery.Client()
    dataset_id = 'project-mlops9-rbarzola.predicciones_dataset'  # Reemplazado con tu dataset
    table_id = 'predictions_table'
    rows_to_insert = [{'prediction': pred} for pred in predictions]
    bq_client.insert_rows_json(f'{dataset_id}.{table_id}', rows_to_insert)
    print("Predicciones almacenadas en BigQuery.")

# Pipeline de Entrenamiento
@dsl.pipeline(name='train-model-pipeline', pipeline_root='gs://vertex-ai-despliegue-en-gcp/pipeline-root')
def train_pipeline():
    train_model = train_model_op()

# Pipeline de Predicción
@dsl.pipeline(name='prediction-pipeline', pipeline_root='gs://vertex-ai-despliegue-en-gcp/pipeline-root')
def prediction_pipeline():
    validate = validate_bigquery_tables_op()
    impute = impute_data_op()
    prediction = prediction_op()
    prediction.after(impute)  # La predicción ocurre después de la imputación

