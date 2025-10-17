#!/usr/bin/env python3
"""
PIPELINE AUTOMATIZADO DE RE-TREINAMENTO
Sistema de Detecção de AML
"""

import json
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
import pickle

class RetrainingPipeline:
    def __init__(self, config_path='config/production_config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.monitor = None  # Integrar com AMLMonitor

    def load_config(self, config_path):
        """Carrega configuração de produção"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Configura logging"""
        logging.basicConfig(
            filename='logs/retraining.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def check_retraining_trigger(self):
        """Verifica se re-treinamento deve ser acionado"""
        triggers = []

        # Trigger 1: Performance degradation
        if self.monitor:
            health_report = self.monitor.get_health_report()
            if health_report['status'] == 'critical':
                triggers.append('performance_degradation')

        # Trigger 2: Data drift
        drift_score = self.calculate_data_drift()
        if drift_score > self.config.get('drift_detection', {}).get('retraining_trigger_threshold', 0.2):
            triggers.append('data_drift')

        # Trigger 3: Scheduled retraining (mensal)
        last_retraining = self.get_last_retraining_date()
        if (datetime.now() - last_retraining).days > 30:
            triggers.append('scheduled_retraining')

        # Trigger 4: New data volume
        new_data_volume = self.get_new_data_volume()
        if new_data_volume > self.config.get('retraining', {}).get('min_new_samples', 10000):
            triggers.append('new_data_available')

        return triggers

    def calculate_data_drift(self):
        """Calcula score de drift dos dados"""
        # Implementação simplificada
        return 0.05  # Placeholder

    def get_last_retraining_date(self):
        """Obtém data do último re-treinamento"""
        try:
            with open('artifacts/last_retraining.json', 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data['timestamp'])
        except:
            return datetime.now() - timedelta(days=60)  # Default para 60 dias atrás

    def get_new_data_volume(self):
        """Obtém volume de novos dados disponíveis"""
        # Implementação simplificada
        return 50000  # Placeholder

    def prepare_training_data(self):
        """Prepara dados para re-treinamento"""
        self.logger.info("Preparando dados de treinamento...")

        try:
            # Carregar dados históricos + novos
            historical_data = pd.read_pickle('artifacts/features_processed.pkl')

            # Simular novos dados (em produção, carregar do data lake)
            new_samples = len(historical_data) // 10  # 10% de novos dados
            new_data = historical_data.sample(new_samples, replace=True)
            new_data['is_fraud'] = np.random.choice([0, 1], new_samples,
                                                  p=[0.998, 0.002])  # Taxa de fraude ligeiramente diferente

            # Combinar datasets
            combined_data = pd.concat([historical_data, new_data], ignore_index=True)

            # Split train/validation
            X = combined_data.drop('is_fraud', axis=1)
            y = combined_data['is_fraud']

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.logger.info(f"Dados preparados: {len(X_train)} treino, {len(X_val)} validação")

            return X_train, X_val, y_train, y_val

        except Exception as e:
            self.logger.error(f"Erro ao preparar dados: {e}")
            return None, None, None, None

    def retrain_models(self, X_train, X_val, y_train, y_val):
        """Re-treina modelos com dados atualizados"""
        self.logger.info("Iniciando re-treinamento dos modelos...")

        from sklearn.ensemble import RandomForestClassifier
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import VotingClassifier

        models = {}

        try:
            # XGBoost
            self.logger.info("Treinando XGBoost...")
            xgb_model = xgb.XGBClassifier(
                max_depth=6, learning_rate=0.1, n_estimators=200,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model

            # LightGBM
            self.logger.info("Treinando LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                num_leaves=31, learning_rate=0.1, n_estimators=150,
                random_state=42
            )
            lgb_model.fit(X_train, y_train)
            models['LightGBM'] = lgb_model

            # RandomForest
            self.logger.info("Treinando RandomForest...")
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model

            # Ensemble
            self.logger.info("Criando ensemble...")
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('rf', rf_model)
                ],
                voting='soft'
            )
            ensemble.fit(X_train, y_train)
            models['Ensemble'] = ensemble

            self.logger.info("Re-treinamento concluído")
            return models

        except Exception as e:
            self.logger.error(f"Erro no re-treinamento: {e}")
            return None

    def validate_retrained_models(self, models, X_val, y_val):
        """Valida performance dos modelos re-treinados"""
        self.logger.info("Validando modelos re-treinados...")

        from sklearn.metrics import precision_recall_curve, auc, f1_score

        validation_results = {}

        for name, model in models.items():
            try:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)

                precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
                pr_auc = auc(recall, precision)
                f1 = f1_score(y_val, y_pred)

                validation_results[name] = {
                    'pr_auc': pr_auc,
                    'f1_score': f1,
                    'validation_samples': len(y_val)
                }

                self.logger.info(f"{name}: PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

            except Exception as e:
                self.logger.error(f"Erro na validação de {name}: {e}")
                validation_results[name] = {'error': str(e)}

        return validation_results

    def deploy_new_models(self, models, validation_results):
        """Faz deploy dos novos modelos se validação passar"""
        self.logger.info("Verificando se modelos podem ser implantados...")

        # Verificar se todos os modelos passaram na validação
        min_pr_auc = self.config.get('model_validation', {}).get('min_pr_auc', 0.2)

        deployable = True
        for name, results in validation_results.items():
            if 'error' in results or results.get('pr_auc', 0) < min_pr_auc:
                self.logger.warning(f"{name} não atende critérios mínimos")
                deployable = False

        if deployable:
            # Salvar novos modelos
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            for name, model in models.items():
                filename = f'artifacts/{name.lower()}_retrained_{timestamp}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Modelo salvo: {filename}")

            # Atualizar ponteiros para modelos ativos
            self.update_active_models(timestamp)

            # Registrar re-treinamento
            self.record_retraining(validation_results, timestamp)

            self.logger.info("Novos modelos implantados com sucesso")
            return True
        else:
            self.logger.warning("Modelos não implantados - validação falhou")
            return False

    def update_active_models(self, timestamp):
        """Atualiza ponteiros para modelos ativos"""
        active_models = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'XGBoost': f'xgboost_retrained_{timestamp}.pkl',
                'LightGBM': f'lightgbm_retrained_{timestamp}.pkl',
                'RandomForest': f'randomforest_retrained_{timestamp}.pkl',
                'Ensemble': f'ensemble_retrained_{timestamp}.pkl'
            }
        }

        with open('artifacts/active_models.json', 'w') as f:
            json.dump(active_models, f, indent=2)

    def record_retraining(self, validation_results, timestamp):
        """Registra evento de re-treinamento"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'retraining',
            'validation_results': validation_results,
            'model_versions': timestamp,
            'status': 'completed'
        }

        with open('artifacts/last_retraining.json', 'w') as f:
            json.dump(record, f, indent=2)

    def run_pipeline(self):
        """Executa pipeline completo de re-treinamento"""
        self.logger.info("=== INICIANDO PIPELINE DE RE-TREINAMENTO ===")

        # Verificar triggers
        triggers = self.check_retraining_trigger()
        if not triggers:
            self.logger.info("Nenhum trigger de re-treinamento ativado")
            return False

        self.logger.info(f"Triggers ativados: {triggers}")

        # Preparar dados
        X_train, X_val, y_train, y_val = self.prepare_training_data()
        if X_train is None:
            return False

        # Re-treinar modelos
        models = self.retrain_models(X_train, X_val, y_train, y_val)
        if not models:
            return False

        # Validar modelos
        validation_results = self.validate_retrained_models(models, X_val, y_val)

        # Deploy se válido
        success = self.deploy_new_models(models, validation_results)

        self.logger.info(f"Pipeline concluído: {'Sucesso' if success else 'Falhou'}")
        return success

# Exemplo de uso
if __name__ == "__main__":
    pipeline = RetrainingPipeline()
    success = pipeline.run_pipeline()
    print(f"Pipeline executado: {'Sucesso' if success else 'Falhou'}")
