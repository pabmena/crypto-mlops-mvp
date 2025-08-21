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
sys.path.append('/opt/airflow/ml/models')
from volatility_lstm import VolatilityLSTM 

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
    print("🔄 Extracting crypto data...")
    
    # Configurar exchange
    exchange = ccxt.binance()
    
    # Lista de símbolos a extraer
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
    
    all_data = {}
    
    for symbol in symbols:
        try:
            # Obtener datos (últimas 1000 horas)
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=1000)
            
            # Convertir a DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            all_data[symbol.replace('/', '_')] = df.to_dict('records')
            
            print(f"✅ Extracted {len(df)} records for {symbol}")
            
        except Exception as e:
            print(f"❌ Error extracting {symbol}: {e}")
    
    # Guardar en MinIO
    minio_client = get_minio_client()
    
    # Crear bucket si no existe
    bucket_name = 'raw-data'
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    
    # Subir datos
    date_str = context['ds']  # YYYY-MM-DD
    file_name = f"crypto_data_{date_str}.json"
    
    # Convertir a JSON y subir
    json_data = json.dumps(all_data, default=str, indent=2)
    
    from io import BytesIO
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
    print("⚙️ Processing crypto data...")
    
    # Obtener path del archivo desde task anterior
    ti = context['task_instance']
    file_path = ti.xcom_pull(task_ids='extract_crypto_data')
    bucket_name, file_name = file_path.split('/', 1)
    
    # Descargar desde MinIO
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
            # Convertir a DataFrame
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calcular features técnicos
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=24).std()
            df['sma_12'] = df['close'].rolling(window=12).mean()
            df['sma_48'] = df['close'].rolling(window=48).mean()
            df['volume_sma'] = df['volume'].rolling(window=24).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bb_upper'] = sma_20 + (std_20 * 2)
            df['bb_lower'] = sma_20 - (std_20 * 2)
            
            # Target: volatilidad futura (24h)
            df['target_volatility'] = df['volatility'].shift(-24)
            
            # Limpiar NaN
            df_clean = df.dropna()
            
            processed_data[symbol] = df_clean.to_dict('records')
            
            print(f"✅ Processed {len(df_clean)} clean records for {symbol}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
    
    # Subir datos procesados
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
    print("🔍 Validating data quality...")
    
    ti = context['task_instance']
    file_path = ti.xcom_pull(task_ids='process_crypto_data')
    bucket_name, file_name = file_path.split('/', 1)
    
    # Descargar datos procesados
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
        
        # Detectar anomalías de precio (cambios > 50%)
        if 'returns' in df.columns:
            extreme_returns = (abs(df['returns']) > 0.5).sum()
            checks['price_anomalies'] = extreme_returns
        
        # Detectar anomalías de volumen (> 10x mediana)
        if 'volume' in df.columns and len(df) > 0:
            median_vol = df['volume'].median()
            if median_vol > 0:
                high_vol = (df['volume'] > median_vol * 10).sum()
                checks['volume_anomalies'] = high_vol
        
        quality_report['quality_checks'][symbol] = checks
        
        # Criterios de fallo
        if (checks['record_count'] < 100 or 
            checks['missing_values'] > checks['record_count'] * 0.1 or
            checks['price_anomalies'] > 10):
            
            quality_report['passed'] = False
            print(f"⚠️ Quality check FAILED for {symbol}: {checks}")
        else:
            print(f"✅ Quality check PASSED for {symbol}")
    
    # Guardar reporte de calidad
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

from ml.models.volatility_lstm import train_model_pipeline, setup_mlflow_environment
import mlflow

def retrain_model(**context):
    try:
        # 1. Configurar el entorno de MLflow
        if not setup_mlflow_environment():
            raise Exception("No se pudo configurar el entorno de MLflow.")

        # 2. Iniciar la ejecución de MLflow desde el DAG (¡SOLO AQUÍ!)
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")

            # 3. Llamar al pipeline de entrenamiento de tu modelo
            #    Ahora, tu función train_model_pipeline YA NO debe tener un with mlflow.start_run()
            trained_model, mse, mae = train_model_pipeline("BTC/USDT", days_back=60)

            # 4. Registrar el modelo en el Model Registry de MLflow
            #    Se usa el URI de la run actual para la versión
            model_info = mlflow.tensorflow.log_model(
                tf_saved_model_dir=trained_model,
                artifact_path="model",
                signature=infer_signature(X_train_seq, y_pred_scaled) # Reemplaza con tus variables
            )
            
            model_version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name="crypto-volatility-lstm"
            )

            print(f"✅ Modelo entrenado y registrado exitosamente como versión: {model_version.version}")

            return {
                'run_id': run.info.run_id,
                'model_version': model_version.version,
            }

    except Exception as e:
        print(f"❌ Error durante el entrenamiento y registro del modelo: {e}")
        # Lanza la excepción para que Airflow marque la tarea como 'Failed'
        raise e

def deploy_model(**context):
    """Task 5: Deploy del modelo a producción"""
    print("🚀 Deploying model to production...")
    
    ti = context['task_instance']
    training_result = ti.xcom_pull(task_ids='retrain_model')
    
    if not training_result:
        print("⚠️ No training result available")
        return False
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    try:
        # Registrar modelo en Model Registry
        model_name = "volatility_lstm"
        run_id = training_result['run_id']
        
        model_uri = f"runs:/{run_id}/model"
        
        # Registrar el modelo en el Model Registry de MLflow
        mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Model deployed successfully. Run ID: {run_id}")
        
        # Notificar a la API que recargue el modelo
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
            'run_id': run_id,
            'deployment_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return {'deployed': False, 'error': str(e)}

# Definir DAG
dag = DAG(
    'crypto_ml_pipeline',
    default_args=default_args,
    description='Pipeline completo de ML para datos crypto',
    schedule_interval=timedelta(hours=6),  # Ejecutar cada 6 horas
    catchup=False,
    tags=['crypto', 'ml', 'etl']
)

# Definir tasks
extract_task = PythonOperator(
    task_id='extract_crypto_data',
    python_callable=extract_crypto_data,
    dag=dag
)

process_task = PythonOperator(
    task_id='process_crypto_data',
    python_callable=process_crypto_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Definir dependencias
extract_task >> process_task >> validate_task >> retrain_task >> deploy_task