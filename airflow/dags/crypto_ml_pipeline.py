"""
DAG de Airflow para pipeline de ML de criptomonedas
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from io import BytesIO
import pandas as pd
import ccxt
import mlflow
import boto3
from minio import Minio
import json
import os
import sys
import random
import time
import tensorflow as tf
from mlflow.models.signature import infer_signature # Necesitas infer_signature

# Configuración
default_args = {
    'owner': 'crypto-mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# La ruta para importar tu modelo
sys.path.append('/opt/airflow')
from ml.models.volatility_lstm import VolatilityLSTM

# Configurar MinIO client
def get_minio_client():
    return Minio(
        'minio:9000',
        access_key='minioadmin',
        secret_key='minioadmin123',
        secure=False
    )

def extract_crypto_data(**context):
    """Task 1: Extraer datos de crypto desde Binance"""
    # Tu código actual está bien, no hay cambios necesarios aquí
    # ... (código de extract_crypto_data)
    print("🔄 Extracting crypto data...")
    exchange = ccxt.binance()
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
    all_data = {}
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            all_data[symbol.replace('/', '_')] = df.to_dict('records')
            print(f"✅ Extracted {len(df)} records for {symbol}")
        except Exception as e:
            print(f"❌ Error extracting {symbol}: {e}")
    minio_client = get_minio_client()
    bucket_name = 'raw-data'
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    date_str = context['ds']
    file_name = f"crypto_data_{date_str}.json"
    json_data = json.dumps(all_data, default=str, indent=2)
    data_stream = BytesIO(json_data.encode('utf-8'))
    minio_client.put_object(
        bucket_name,
        file_name,
        data_stream,
        length=len(json_data),
        content_type='application/json'
    )
    print(f"📁 Data uploaded to MinIO: {bucket_name}/{file_name}")
    return f"{bucket_name}/{file_name}"

def process_crypto_data(**context):
    """Task 2: Procesar y limpiar datos"""
    # Tu código actual está bien, no hay cambios necesarios aquí
    # ... (código de process_crypto_data)
    print("⚙️ Processing crypto data...")
    ti = context['task_instance']
    file_path = ti.xcom_pull(task_ids='extract_crypto_data')
    bucket_name, file_name = file_path.split('/', 1)
    minio_client = get_minio_client()
    try:
        response = minio_client.get_object(bucket_name, file_name)
        raw_data = json.loads(response.read().decode('utf-8'))
        response.close()
        response.release_conn()
        print(f"📥 Downloaded data from MinIO: {len(raw_data)} symbols")
    except Exception as e:
        print(f"❌ Error downloading from MinIO: {e}")
        raise
    processed_data = {}
    for symbol, records in raw_data.items():
        try:
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=24).std()
            df['sma_12'] = df['close'].rolling(window=12).mean()
            df['sma_48'] = df['close'].rolling(window=48).mean()
            df['volume_sma'] = df['volume'].rolling(window=24).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            df['target_volatility'] = df['volatility'].shift(-24)
            df_clean = df.dropna()
            processed_data[symbol] = df_clean.to_dict('records')
            print(f"✅ Processed {len(df_clean)} clean records for {symbol}")
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    processed_bucket = 'processed-data'
    if not minio_client.bucket_exists(processed_bucket):
        minio_client.make_bucket(processed_bucket)
    date_str = context['ds']
    processed_file = f"processed_crypto_{date_str}.json"
    json_data = json.dumps(processed_data, default=str, indent=2)
    data_stream = BytesIO(json_data.encode('utf-8'))
    minio_client.put_object(
        processed_bucket,
        processed_file,
        data_stream,
        length=len(json_data),
        content_type='application/json'
    )
    print(f"📁 Processed data uploaded: {processed_bucket}/{processed_file}")
    return f"{processed_bucket}/{processed_file}"

def validate_data_quality(**context):
    """Task 3: Validar calidad de datos"""
    # Tu código actual está bien, no hay cambios necesarios aquí
    # ... (código de validate_data_quality)
    print("🔍 Validating data quality...")
    ti = context['task_instance']
    file_path = ti.xcom_pull(task_ids='process_crypto_data')
    bucket_name, file_name = file_path.split('/', 1)
    minio_client = get_minio_client()
    response = minio_client.get_object(bucket_name, file_name)
    processed_data = json.loads(response.read().decode('utf-8'))
    response.close()
    response.release_conn()
    quality_report = {
        'timestamp': datetime.now().isoformat(),
        'date': context['ds'],
        'symbols_processed': len(processed_data),
        'quality_checks': {},
        'passed': True
    }
    for symbol, records in processed_data.items():
        df = pd.DataFrame(records)
        checks = {
            'record_count': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_timestamps': df.duplicated(subset=['timestamp']).sum(),
            'price_anomalies': 0,
            'volume_anomalies': 0
        }
        if 'returns' in df.columns:
            extreme_returns = (abs(df['returns']) > 0.5).sum()
            checks['price_anomalies'] = extreme_returns
        if 'volume' in df.columns and len(df) > 0:
            median_vol = df['volume'].median()
            if median_vol > 0:
                high_vol = (df['volume'] > median_vol * 10).sum()
                checks['volume_anomalies'] = high_vol
        quality_report['quality_checks'][symbol] = checks
        if (checks['record_count'] < 100 or
            checks['missing_values'] > checks['record_count'] * 0.1 or
            checks['price_anomalies'] > 10):
            quality_report['passed'] = False
            print(f"⚠️ Quality check FAILED for {symbol}: {checks}")
        else:
            print(f"✅ Quality check PASSED for {symbol}")
    reports_bucket = 'quality-reports'
    if not minio_client.bucket_exists(reports_bucket):
        minio_client.make_bucket(reports_bucket)
    date_str = context['ds']
    report_file = f"quality_report_{date_str}.json"
    json_data = json.dumps(quality_report, default=str, indent=2)
    data_stream = BytesIO(json_data.encode('utf-8'))
    minio_client.put_object(
        reports_bucket,
        report_file,
        data_stream,
        length=len(json_data),
        content_type='application/json'
    )
    print(f"📊 Quality report saved: {reports_bucket}/{report_file}")
    if not quality_report['passed']:
        raise ValueError("Data quality validation failed!")
    return quality_report

def retrain_model(**context):
    """Task 4: Re-entrenar modelo de ML"""
    print("🤖 Retraining ML model...")
    try:
        # 1. Obtener los datos procesados desde MinIO
        ti = context['task_instance']
        file_path = ti.xcom_pull(task_ids='process_crypto_data')
        bucket_name, file_name = file_path.split('/', 1)

        minio_client = get_minio_client()
        response = minio_client.get_object(bucket_name, file_name)
        processed_data = json.loads(response.read().decode('utf-8'))
        response.close()
        response.release_conn()

        # 2. Convertir los datos a un DataFrame de Pandas
        df = pd.DataFrame(list(processed_data.values())[0])
        print(f"📊 Training data shape: {df.shape}")

        # 3. Iniciar la ejecución de MLflow desde el DAG (¡SOLO AQUÍ!)
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")

            # 4. Instanciar tu modelo y entrenarlo
            model_instance = VolatilityLSTM(sequence_length=48, lstm_units=64, dropout_rate=0.2)

            # Llama al método de entrenamiento de tu clase.
            # Este método ahora maneja todo el registro de MLflow internamente.
            # No devuelve un objeto con 'model_uri' porque no es necesario.
            model_instance.train(
                df,
                epochs=30, 
                batch_size=16, 
                registered_model_name="crypto-predictor"
            )

            # 5. La tarea del DAG simplemente devuelve el ID de la ejecución de MLflow
            return {
                'run_id': run.info.run_id
            }

    except Exception as e:
        print(f"❌ Error durante el entrenamiento y registro del modelo: {e}")
        # Lanza la excepción para que Airflow marque la tarea como 'Failed'
        raise e

def deploy_model(**context):
    """Task 5: Desplegar el modelo a producción"""
    print("🚀 Deploying model to production...")

    ti = context['task_instance']
    training_result = ti.xcom_pull(task_ids='retrain_model')

    if not training_result:
        print("⚠️ No training result available")
        return False
    
    # 1. Conectar al servidor de MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # 2. Cargar el cliente del Model Registry
    client = mlflow.tracking.MlflowClient()

    # 3. Obtener la última versión del modelo registrado
    model_name = "crypto-predictor"
    versions = client.get_latest_versions(model_name, ["None"])
    if not versions:
        print(f"⚠️ No se encontró ninguna versión del modelo '{model_name}'.")
        return False
    
    model_version = versions[0]
    
    # 4. Transicionar la versión a la etapa de "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    print(f"✅ Versión {model_version.version} del modelo '{model_name}' transicionada a Production.")

    # 5. Notificar a la API que recargue el modelo
    import requests
    try:
        response = requests.post("http://api:8800/v1/ml/model/reload")
        if response.status_code == 200:
            print("✅ API model reloaded successfully")
        else:
            print(f"⚠️ API reload failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not notify API: {e}")
    
    return {
        'deployed': True,
        'model_name': model_name,
        'model_version': model_version.version,
        'deployment_time': datetime.now().isoformat()
    }
    
# Definir DAG
dag = DAG(
    'crypto_ml_pipeline',
    default_args=default_args,
    description='Pipeline completo de ML para datos crypto',
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=['crypto', 'ml', 'etl']
)

# Definir tasks
extract_task = PythonOperator(task_id='extract_crypto_data', python_callable=extract_crypto_data, dag=dag)
process_task = PythonOperator(task_id='process_crypto_data', python_callable=process_crypto_data, dag=dag)
validate_task = PythonOperator(task_id='validate_data_quality', python_callable=validate_data_quality, dag=dag)
retrain_task = PythonOperator(task_id='retrain_model', python_callable=retrain_model, dag=dag)
deploy_task = PythonOperator(task_id='deploy_model', python_callable=deploy_model, dag=dag)

# Definir dependencias
extract_task >> process_task >> validate_task >> retrain_task >> deploy_task