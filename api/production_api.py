#!/usr/bin/env python3
"""
API DE PRODUÇÃO PARA SISTEMA AML
Inclui monitoramento e health checks integrados
"""

from flask import Flask, request, jsonify
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pathlib import Path

# Importar serviços de monitoramento
from src.monitoring_service import AMLMonitor

app = Flask(__name__)

# Configuração global
MODEL_PATHS = {
    'XGBoost': 'artifacts/xgboost_extended.pkl',
    'LightGBM': 'artifacts/lightgbm_extended.pkl',
    'RandomForest': 'artifacts/randomforest_extended.pkl',
    'Ensemble': 'artifacts/ensemble_extended.pkl'
}

# Carregar modelos
models = {}
for name, path in MODEL_PATHS.items():
    try:
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"✅ {name} carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar {name}: {e}")

# Inicializar monitor
monitor = AMLMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check"""
    health_report = monitor.get_health_report()

    status_code = 200
    if health_report['status'] == 'critical':
        status_code = 503
    elif health_report['status'] == 'warning':
        status_code = 200  # Still operational

    return jsonify({
        'status': health_report['status'],
        'timestamp': datetime.now().isoformat(),
        'version': '2.0',
        'models_loaded': list(models.keys()),
        'metrics': health_report.get('metrics_summary', {})
    }), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de predição"""
    start_time = time.time()

    try:
        # Validar input
        data = request.get_json()
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Invalid input format'}), 400

        transactions = data['transactions']
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400

        # Converter para DataFrame
        df = pd.DataFrame(transactions)

        # Validar features necessárias (simplificado)
        required_features = ['amount', 'payment_format', 'hour', 'is_foreign']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        # Fazer predições
        predictions = {}

        for model_name, model in models.items():
            try:
                pred_proba = model.predict_proba(df)[:, 1]
                predictions[model_name] = {
                    'probabilities': pred_proba.tolist(),
                    'predictions': (pred_proba > 0.5).astype(int).tolist()
                }
            except Exception as e:
                predictions[model_name] = {'error': str(e)}

        # Calcular latência
        latency_ms = (time.time() - start_time) * 1000

        # Simular ground truth para monitoramento (em produção, coletar real)
        y_true_simulated = np.random.choice([0, 1], len(df), p=[0.998, 0.002])

        # Coletar métricas para monitoramento
        ensemble_proba = np.array(predictions.get('Ensemble', {}).get('probabilities', [0] * len(df)))
        if len(ensemble_proba) == len(df):
            metrics = monitor.collect_metrics(y_true_simulated, ensemble_proba, df, latency_ms)

            # Verificar alertas
            alerts = monitor.check_alerts(metrics)
            for alert in alerts:
                monitor.send_alert(alert)

        response = {
            'predictions': predictions,
            'metadata': {
                'model_version': '2.0',
                'timestamp': datetime.now().isoformat(),
                'latency_ms': latency_ms,
                'num_transactions': len(df)
            }
        }

        # Log da requisição
        app.logger.info(f"Predição realizada: {len(df)} transações, latência: {latency_ms:.2f}ms")

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint para consultar métricas de monitoramento"""
    try:
        # Retornar últimas 100 medições
        recent_metrics = monitor.metrics_history[-100:] if monitor.metrics_history else []

        return jsonify({
            'metrics': recent_metrics,
            'total_measurements': len(monitor.metrics_history),
            'alerts_today': len([a for a in monitor.alert_history
                               if a['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def trigger_retraining():
    """Endpoint para acionar re-treinamento (admin only)"""
    try:
        # Em produção, adicionar autenticação
        from scripts.retraining_pipeline import RetrainingPipeline

        pipeline = RetrainingPipeline()
        success = pipeline.run_pipeline()

        return jsonify({
            'status': 'success' if success else 'failed',
            'message': 'Retraining pipeline executed',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        app.logger.error(f"Erro no re-treinamento: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    print("🚀 Iniciando API de produção AML...")
    print("📊 Modelos carregados:", list(models.keys()))
    print("🔍 Monitoramento ativo")

    app.run(host='0.0.0.0', port=8000, debug=False)
