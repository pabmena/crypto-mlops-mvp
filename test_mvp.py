#!/usr/bin/env python3
"""
Script de prueba para el Crypto MLOps MVP
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8800"

def test_health():
    """Probar health check"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health OK - ML Available: {data.get('ml_available', False)}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_crypto_data():
    """Probar obtención de datos crypto"""
    print("\n📊 Testing crypto data...")
    try:
        response = requests.get(f"{BASE_URL}/v1/crypto/ohlcv?symbol=BTCUSDT&limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got {data['rows']} rows of {data['symbol']} data")
            print(f"   Latest price: ${data['data'][-1]['close']:,.2f}")
            return True
        else:
            print(f"❌ Crypto data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Crypto data error: {e}")
        return False

def test_heuristic_signal():
    """Probar señal heurística"""
    print("\n🧠 Testing heuristic signal...")
    try:
        payload = {
            "symbol": "BTCUSDT",
            "horizon_min": 60,
            "explain": True
        }
        response = requests.post(f"{BASE_URL}/v1/crypto/signal", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Heuristic signal generated")
            print(f"   Risk Score: {data['risk_score']:.3f}")
            print(f"   Vol Regime: {data['vol_regime']}")
            print(f"   Nowcast Return: {data['nowcast_ret']:.4f}")
            return True
        else:
            print(f"❌ Heuristic signal failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Heuristic signal error: {e}")
        return False

def test_ml_signal():
    """Probar señal ML"""
    print("\n🤖 Testing ML signal...")
    try:
        payload = {
            "symbol": "BTCUSDT",
            "include_heuristic": True
        }
        response = requests.post(f"{BASE_URL}/v1/crypto/ml-signal", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ML signal generated")
            print(f"   Model Version: {data.get('model_version', 'N/A')}")
            ml_pred = data.get('ml_prediction', {})
            print(f"   ML Prediction: {ml_pred.get('prediction', 'N/A')}")
            print(f"   Confidence: {ml_pred.get('confidence', 'N/A')}")
            return True
        elif response.status_code == 503:
            print("⚠️ ML service not available (expected if no model trained yet)")
            return True
        else:
            print(f"❌ ML signal failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ML signal error: {e}")
        return False

def test_signals_comparison():
    """Probar comparación de señales"""
    print("\n⚖️ Testing signals comparison...")
    try:
        response = requests.get(f"{BASE_URL}/v1/crypto/signals/compare?symbol=BTCUSDT")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Signals comparison generated")
            heur = data.get('heuristic', {})
            ml = data.get('ml', {})
            comp = data.get('comparison', {})
            print(f"   Heuristic Vol: {heur.get('volatility', 'N/A')}")
            print(f"   ML Vol: {ml.get('predicted_volatility', 'N/A')}")
            print(f"   Regime Agreement: {comp.get('regime_agreement', 'N/A')}")
            return True
        elif response.status_code == 503:
            print("⚠️ ML service not available for comparison")
            return True
        else:
            print(f"❌ Signals comparison failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Signals comparison error: {e}")
        return False

def test_recent_signals():
    """Probar obtención de señales recientes"""
    print("\n📈 Testing recent signals...")
    try:
        response = requests.get(f"{BASE_URL}/v1/crypto/signals/tail?n=3")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got {len(data)} recent signals")
            for i, signal in enumerate(data[:2]):  # Show first 2
                timestamp = signal.get('timestamp', 'N/A')
                method = signal.get('method', 'N/A')
                print(f"   Signal {i+1}: {method} at {timestamp}")
            return True
        else:
            print(f"❌ Recent signals failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Recent signals error: {e}")
        return False

def main():
    """Ejecutar todas las pruebas"""
    print("🚀 Crypto MLOps MVP - Test Suite")
    print("=" * 50)
    
    tests = [
        test_health,
        test_crypto_data,
        test_heuristic_signal,
        test_ml_signal,
        test_signals_comparison,
        test_recent_signals
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Pequeña pausa entre tests
        except KeyboardInterrupt:
            print("\n⚠️ Tests interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Your MVP is working correctly!")
    elif passed >= total - 1:
        print("✅ Most tests passed! MVP is mostly functional!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n🌐 Access your services at:")
    print(f"   FastAPI Docs: http://localhost:8800/docs")
    print(f"   MLFlow: http://localhost:5000")
    print(f"   Airflow: http://localhost:8080")
    print(f"   MinIO: http://localhost:9001")

if __name__ == "__main__":
    main()