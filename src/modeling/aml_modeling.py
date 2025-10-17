"""
AML Modeling Module
Funções para treinamento, validação e avaliação de modelos AML com foco em compliance regulatória.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    recall_score, precision_score, f1_score, confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Model imports
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available - install with: pip install shap")

# Live Plot Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time

# Plotly para plots interativos em tempo real
try:
    import plotly.graph_objects as go
    from plotly.graph_objects import FigureWidget
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available - using matplotlib for live plots")

class LivePlotCallback:
    """
    Callback para plotar métricas de treinamento em tempo real.
    """

    def __init__(self, model_name, update_interval=10, use_plotly=True):
        self.model_name = model_name
        self.update_interval = update_interval
        self.iterations = []
        self.train_auc = []
        self.val_auc = []
        self.oob_scores = []
        self.start_time = time.time()
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE

        if self.use_plotly:
            self._init_plotly_plot()
        else:
            print("Using matplotlib for live plot")

    def _init_plotly_plot(self):
        """Inicializa plot plotly interativo usando FigureWidget."""
        # Usar FigureWidget para atualização em tempo real
        self.fig = FigureWidget(make_subplots(
            rows=1, cols=2,
            subplot_titles=('Métricas de Treinamento', 'Progresso'),
            specs=[[{"secondary_y": True}, {"type": "indicator"}]]
        ))

        # Adicionar traces vazias
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='AUC Validação',
                                    line=dict(color='blue', width=3)), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='AUC Treino',
                                    line=dict(color='red', dash='dash')), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='OOB Score',
                                    line=dict(color='green', width=2)), row=1, col=1, secondary_y=True)

        # Indicador de progresso
        self.fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Iteração"},
            gauge={'axis': {'range': [0, 100]}},
            domain={'row': 0, 'column': 1}
        ), row=1, col=2)

        self.fig.update_layout(
            height=400,
            showlegend=True,
            title_text=f"{self.model_name.upper()} - Treinamento em Tempo Real"
        )

        self.fig.update_xaxes(title_text="Iterações", row=1, col=1)
        self.fig.update_yaxes(title_text="AUC", row=1, col=1)
        self.fig.update_yaxes(title_text="OOB Score", secondary_y=True, row=1, col=1)

        # Display apenas uma vez - o widget será atualizado
        from IPython.display import display
        display(self.fig)

    def update_plot(self):
        """Atualiza o plot em tempo real."""
        if self.use_plotly:
            self._update_plotly_plot()
        else:
            self._update_matplotlib_plot()

    def _update_plotly_plot(self):
        """Atualiza plot plotly diretamente nos dados."""
        # Atualizar dados diretamente (FigureWidget permite isso)
        self.fig.data[0].x = self.iterations
        self.fig.data[0].y = self.val_auc if self.val_auc else []

        self.fig.data[1].x = self.iterations
        self.fig.data[1].y = self.train_auc if self.train_auc else []

        self.fig.data[2].x = self.iterations
        self.fig.data[2].y = self.oob_scores if self.oob_scores else []

        # Atualizar indicador de progresso
        current_iter = self.iterations[-1] if self.iterations else 0
        self.fig.data[3].value = current_iter % 100  # Ciclo a cada 100 iterações

        # Atualizar título com tempo
        elapsed = time.time() - self.start_time
        self.fig.update_layout(
            title_text=f"{self.model_name.upper()} - Treinamento em Tempo Real (Tempo: {elapsed:.1f}s)"
        )

    def _update_matplotlib_plot(self):
        """Atualiza plot matplotlib (fallback)."""
        clear_output(wait=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if self.val_auc:
            ax.plot(self.iterations[:len(self.val_auc)], self.val_auc, 'b-', label='AUC Validação', linewidth=2)
        if self.train_auc:
            ax.plot(self.iterations[:len(self.train_auc)], self.train_auc, 'r--', label='AUC Treino', alpha=0.7)
        if self.oob_scores:
            ax.plot(self.iterations[:len(self.oob_scores)], self.oob_scores, 'g-', label='OOB Score', linewidth=2)

        ax.set_title(f'{self.model_name.upper()} - Treinamento em Tempo Real', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iterações')
        ax.set_ylabel('Métrica')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Informações de progresso
        elapsed = time.time() - self.start_time
        if self.iterations:
            current_iter = self.iterations[-1]
            ax.text(0.02, 0.98, f'Iteração: {current_iter}\nTempo: {elapsed:.1f}s',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def add_metrics(self, iteration, train_auc=None, val_auc=None, oob_score=None):
        """Adiciona métricas e atualiza plot se necessário."""
        self.iterations.append(iteration)
        if train_auc is not None:
            self.train_auc.append(train_auc)
        if val_auc is not None:
            self.val_auc.append(val_auc)
        if oob_score is not None:
            self.oob_scores.append(oob_score)

        if iteration % self.update_interval == 0 or iteration == 1:
            self.update_plot()

    def finalize_plot(self):
        """Finaliza o plot com métricas finais."""
        if self.use_plotly:
            # Adicionar anotações finais diretamente no FigureWidget
            final_text = f"Treinamento Concluído!<br>Iterações: {len(self.iterations)}<br>Tempo: {time.time() - self.start_time:.1f}s"
            self.fig.add_annotation(
                text=final_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                bgcolor="lightgreen", bordercolor="black", borderwidth=1
            )
            # O FigureWidget já está sendo exibido, não precisa de display adicional

class XGBoostLivePlotCallback(xgb.callback.TrainingCallback):
    """Callback XGBoost para plot em tempo real."""

    def __init__(self, live_plot_callback):
        self.live_plot = live_plot_callback
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        """Chamado após cada iteração."""
        if evals_log:
            # Extrair AUC de validação
            val_auc = None
            train_auc = None
            if 'validation' in evals_log and 'auc' in evals_log['validation']:
                val_auc = evals_log['validation']['auc'][-1]
            if 'train' in evals_log and 'auc' in evals_log['train']:
                train_auc = evals_log['train']['auc'][-1]

            self.live_plot.add_metrics(epoch + 1, train_auc=train_auc, val_auc=val_auc)
        return False  # Continuar treinamento


class AMLModelTrainer:
    """
    Trainer especializado para modelos AML com validação temporal e compliance regulatória.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o trainer com configuração experimental.

        Args:
            config: Dicionário com parâmetros experimentais
        """
        self.config = config
        self.trained_models = {}
        self.cv_results = {}
        self.feature_importance = {}
        self.training_history = {}  # Histórico de métricas de treinamento para visualização

    def _get_model_class(self, model_type: str):
        """Retorna a classe do modelo baseada no tipo."""
        model_classes = {
            'xgb': XGBClassifier,
            'lgb': LGBMClassifier,
            'rf': RandomForestClassifier
        }
        return model_classes.get(model_type)

    def build_pipeline(self, model_name: str) -> Pipeline:
        """
        Constrói pipeline sklearn para um modelo específico.

        Args:
            model_name: Nome do modelo (xgboost, lightgbm, random_forest)

        Returns:
            Pipeline configurado
        """
        if model_name not in self.config['models']:
            raise ValueError(f"Modelo {model_name} não encontrado na configuração")

        model_config = self.config['models'][model_name]
        model_class = self._get_model_class(model_config['model_type'])

        if model_class is None:
            raise ValueError(f"Tipo de modelo {model_config['model_type']} não suportado")

        # Criar modelo com parâmetros
        model = model_class(**model_config['params'])

        # Pipeline com scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        return pipeline

    def train_with_temporal_cv(self, X: pd.DataFrame, y: pd.Series,
                             model_name: str, live_plot: LivePlotCallback = None,
                             streamlit_callback: 'StreamlitLiveCallback' = None) -> Dict[str, Any]:
        """
        Treina modelo com validação cruzada temporal e early stopping para evitar data leakage.

        Args:
            X: Features
            y: Target
            model_name: Nome do modelo

        Returns:
            Dicionário com resultados do treinamento
        """
        # Construir pipeline
        pipeline = self.build_pipeline(model_name)

        # Validação cruzada temporal - DESABILITAR early stopping temporariamente
        cv_results = {}
        original_early_stopping = self.config['early_stopping']['enabled']

        # Criar TimeSeriesSplit para validação temporal
        tscv = TimeSeriesSplit(n_splits=self.config['temporal_splits'])

        try:
            # Desabilitar early stopping para CV temporal
            self.config['early_stopping']['enabled'] = False

            # Usar métricas da configuração
            scoring_metrics = self.config['metrics']

            for metric in scoring_metrics:
                # Pular pr_auc pois é calculada manualmente
                if metric == 'pr_auc':
                    continue
                scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=metric)
                cv_results[metric] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std()
                }
        finally:
            # Restaurar configuração original
            self.config['early_stopping']['enabled'] = original_early_stopping

        # Treinar modelo final no conjunto completo com early stopping se disponível
        pipeline = self._train_with_early_stopping(pipeline, X, y, model_name, live_plot, streamlit_callback)

        # Finalizar plot se callback foi usado
        if live_plot:
            live_plot.finalize_plot()

        # Armazenar resultados
        results = {
            'pipeline': pipeline,
            'cv_results': cv_results,
            'best_score': cv_results['roc_auc']['mean'],
            'training_time': datetime.now().isoformat(),
            'model_name': model_name,
            'early_stopping_info': getattr(pipeline.named_steps['model'], 'best_iteration', None)
        }

        self.trained_models[model_name] = pipeline
        self.cv_results[model_name] = cv_results

        return results

    def _train_with_early_stopping(self, pipeline: Pipeline, X: pd.DataFrame,
                                 y: pd.Series, model_name: str, live_plot: LivePlotCallback = None,
                                 streamlit_callback: 'StreamlitLiveCallback' = None) -> Pipeline:
        """
        Treina modelo com early stopping quando disponível e coleta histórico para visualização.

        Args:
            pipeline: Pipeline sklearn
            X, y: Dados de treinamento
            model_name: Nome do modelo

        Returns:
            Pipeline treinado
        """
        model_config = self.config['models'][model_name]
        model = pipeline.named_steps['model']

        # Inicializar histórico de treinamento
        self.training_history[model_name] = {
            'iterations': [],
            'train_auc': [],
            'val_auc': [],
            'best_iteration': None
        }

        # Verificar se early stopping está habilitado
        if not self.config.get('early_stopping', {}).get('enabled', False):
            pipeline.fit(X, y)
            return pipeline

        es_config = self.config['early_stopping']

        # Early stopping para XGBoost
        if model_config['model_type'] == 'xgb':
            try:
                from sklearn.model_selection import train_test_split

                # Dividir dados para validação
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.config['random_seed'],
                    stratify=y
                )

                # Ajustar scaler nos dados de treino
                pipeline.named_steps['scaler'].fit(X_train)

                # Transformar dados
                X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
                X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)

                # Configurar modelo XGBoost com early stopping
                model = pipeline.named_steps['model']
                model.set_params(
                    early_stopping_rounds=es_config['rounds'],
                    eval_metric=es_config['metric']
                )

                # Preparar callbacks
                callbacks = []
                if live_plot:
                    callbacks.append(XGBoostLivePlotCallback(live_plot))

                # Treinar modelo XGBoost diretamente
                if callbacks:
                    # Se temos callbacks, tentar usar a API antiga
                    try:
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                            callbacks=callbacks,
                            verbose=False
                        )
                    except TypeError:
                        # API mais antiga não suporta callbacks no fit
                        print("   Warning: XGBoost old API - simulating live plot with manual updates.")
                        # Treinar sem callbacks primeiro
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                            verbose=False
                        )

                        # Simular updates do live plot usando evals_result_
                        if live_plot and hasattr(model, 'evals_result_'):
                            eval_result = model.evals_result_
                            # XGBoost nomeia automaticamente como validation_0, validation_1, etc.
                            if 'validation_1' in eval_result and es_config['metric'] in eval_result['validation_1']:
                                val_auc = eval_result['validation_1'][es_config['metric']]  # validation_1 é o conjunto de validação
                                train_auc = eval_result.get('validation_0', {}).get(es_config['metric'], []) if 'validation_0' in eval_result else []

                                # Simular updates progressivos
                                for i, (auc_val, auc_train) in enumerate(zip(val_auc, train_auc if train_auc else [None]*len(val_auc))):
                                    if (i+1) % live_plot.update_interval == 0 or i == len(val_auc) - 1:
                                        live_plot.add_metrics(i+1, train_auc=auc_train, val_auc=auc_val)
                                        if streamlit_callback:
                                            streamlit_callback.add_metrics(i+1, train_auc=auc_train, val_auc=auc_val)
                                        # Pequena pausa para visualização
                                        import time
                                        time.sleep(0.1)
                else:
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                        verbose=False
                    )

                # Atualizar pipeline com modelo treinado
                pipeline.named_steps['model'] = model

                # Armazenar histórico para visualização
                if hasattr(model, 'evals_result_') and 'validation' in model.evals_result_ and es_config['metric'] in model.evals_result_['validation']:
                    self.training_history[model_name]['iterations'] = list(range(1, len(model.evals_result_['validation'][es_config['metric']]) + 1))
                    self.training_history[model_name]['val_auc'] = model.evals_result_['validation'][es_config['metric']]
                    if 'train' in model.evals_result_ and es_config['metric'] in model.evals_result_['train']:
                        self.training_history[model_name]['train_auc'] = model.evals_result_['train'][es_config['metric']]

                if hasattr(model, 'best_iteration'):
                    self.training_history[model_name]['best_iteration'] = model.best_iteration

            except Exception as e:
                print(f"   Error in XGBoost early stopping: {e}. Using standard training.")
                # Remover early_stopping_rounds para evitar erro
                model.set_params(early_stopping_rounds=None)
                pipeline.fit(X, y)

        # Early stopping para LightGBM
        elif model_config['model_type'] == 'lgb':
            try:
                from sklearn.model_selection import train_test_split
                import warnings

                # Suprimir warnings específicos do LightGBM durante treinamento
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

                    # Dividir dados para validação (LightGBM precisa de dados de validação para early stopping)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=self.config['random_seed'],
                        stratify=y
                    )

                    # Ajustar scaler nos dados de treino
                    pipeline.named_steps['scaler'].fit(X_train)

                    # Transformar dados
                    X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
                    X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)

                    # Configurar modelo LightGBM com early stopping
                    model = pipeline.named_steps['model']

                    # Verificar se LightGBM está disponível e importar callbacks
                    try:
                        from lightgbm import early_stopping, log_evaluation
                        lgb_available = True
                    except ImportError:
                        print("   Warning: LightGBM not available. Install with: pip install lightgbm")
                        lgb_available = False

                    if lgb_available:
                        # Configurar callbacks do LightGBM
                        callbacks = [
                            early_stopping(stopping_rounds=es_config['rounds'],
                                         first_metric_only=True,
                                         verbose=False),  # Silenciar early stopping
                            log_evaluation(period=0)  # Completamente silencioso
                        ]

                        # Treinar modelo LightGBM diretamente
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                            eval_names=['train', 'valid'],
                            eval_metric='auc',
                            callbacks=callbacks
                        )

                        # Atualizar pipeline com modelo treinado
                        pipeline.named_steps['model'] = model

                        # Armazenar histórico para visualização
                        if hasattr(model, 'evals_result_'):
                            eval_result = model.evals_result_
                            if 'valid' in eval_result and 'auc' in eval_result['valid']:
                                self.training_history[model_name]['iterations'] = list(range(1, len(eval_result['valid']['auc']) + 1))
                                self.training_history[model_name]['val_auc'] = eval_result['valid']['auc']
                                if 'train' in eval_result and 'auc' in eval_result['train']:
                                    self.training_history[model_name]['train_auc'] = eval_result['train']['auc']

                        if hasattr(model, 'best_iteration_'):
                            self.training_history[model_name]['best_iteration'] = model.best_iteration_
                    else:
                        # Fallback se LightGBM não estiver disponível
                        print("   Using LightGBM training without early stopping")
                        pipeline.fit(X, y)

            except Exception as e:
                print(f"   Error in LightGBM early stopping: {e}. Using standard training.")
                # Remover parâmetros problemáticos e tentar novamente
                try:
                    # Resetar parâmetros que podem causar problemas
                    model.set_params(early_stopping_round=None, callbacks=None)
                    pipeline.fit(X, y)
                    print("   LightGBM trained with simplified configuration")
                except Exception as e2:
                    print(f"   Critical failure in LightGBM training: {e2}")
                    print("   Possible solutions:")
                    print("      - Check installation: pip install lightgbm")
                    print("      - Check data: X.shape, y.shape, data types")
                    print("      - Try reducing model complexity")
                    raise

        # Random Forest com visualização de progresso
        else:
            # Habilitar OOB score para visualização
            model.set_params(oob_score=True, warm_start=True)

            # Treinar incrementalmente para visualização
            n_estimators = model_config['params']['n_estimators']
            step = max(1, n_estimators // 20)  # Atualizar a cada 5% do progresso

            self.training_history[model_name]['iterations'] = []
            self.training_history[model_name]['oob_score'] = []

            for i in range(0, n_estimators, step):
                current_n = min(i + step, n_estimators)
                model.set_params(n_estimators=current_n)
                pipeline.fit(X, y)

                if hasattr(model, 'oob_score_'):
                    self.training_history[model_name]['iterations'].append(current_n)
                    self.training_history[model_name]['oob_score'].append(model.oob_score_)

                    # Atualizar callbacks
                    if live_plot:
                        live_plot.add_metrics(current_n, oob_score=model.oob_score_)
                    if streamlit_callback:
                        streamlit_callback.add_metrics(current_n, oob_score=model.oob_score_)

        return pipeline

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Avalia modelo com métricas focadas em AML.

        Args:
            X: Features
            y: Target
            model_name: Nome do modelo

        Returns:
            Dicionário com métricas de avaliação
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")

        pipeline = self.trained_models[model_name]

        # Predições
        y_pred_proba = pipeline.predict_proba(X)[:, 1]

        # Curvas ROC e PR
        fpr, tpr, roc_thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, pr_thresholds = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)

        # Análise de thresholds AML
        aml_thresholds = self.config['aml_thresholds']
        threshold_analysis = []

        for threshold in aml_thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            threshold_analysis.append({
                'threshold': threshold,
                'recall': recall_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'predicted_fraud_rate': y_pred.mean()
            })

        threshold_df = pd.DataFrame(threshold_analysis)

        # Threshold ótimo (máximo F1)
        optimal_threshold = threshold_df.loc[threshold_df['f1'].idxmax(), 'threshold']

        # Matriz de confusão no threshold ótimo
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm = confusion_matrix(y, y_pred_optimal)

        # Análise de custo-benefício
        cost_config = self.config['business_metrics']['cost_benefit_ratio']
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fp * cost_config['fp_cost']) + (fn * cost_config['fn_cost'])
        cost_per_transaction = total_cost / len(y)

        # Compliance regulatório
        regulatory_req = self.config['business_metrics']['regulatory_requirements']
        final_metrics = threshold_df[threshold_df['threshold'] == optimal_threshold].iloc[0]

        compliance = {
            'recall_ok': final_metrics['recall'] >= regulatory_req['min_recall'],
            'fp_rate_ok': final_metrics['predicted_fraud_rate'] <= regulatory_req['max_false_positive_rate'],
            'overall_compliant': (final_metrics['recall'] >= regulatory_req['min_recall'] and
                                final_metrics['predicted_fraud_rate'] <= regulatory_req['max_false_positive_rate'])
        }

        results = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold,
            'threshold_analysis': threshold_df,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'cost_analysis': {
                'total_cost': total_cost,
                'cost_per_transaction': cost_per_transaction,
                'fp_cost': fp * cost_config['fp_cost'],
                'fn_cost': fn * cost_config['fn_cost']
            },
            'regulatory_compliance': compliance,
            'classification_report': classification_report(y, y_pred_optimal,
                                                         target_names=['Normal', 'Fraude'],
                                                         output_dict=True)
        }

        return results

    def get_feature_importance(self, X: pd.DataFrame, model_name: str,
                             top_n: int = 20) -> pd.DataFrame:
        """
        Extrai importância das features do modelo.

        Args:
            X: Features
            model_name: Nome do modelo
            top_n: Número de top features

        Returns:
            DataFrame com importância das features
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")

        pipeline = self.trained_models[model_name]
        model = pipeline.named_steps['model']

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X.columns

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'importance_pct': importance / importance.sum() * 100
            }).sort_values('importance', ascending=False).head(top_n)

            self.feature_importance[model_name] = importance_df
            return importance_df
        else:
            raise ValueError(f"Modelo {model_name} não suporta análise de feature importance")

    def explain_predictions(self, X: pd.DataFrame, model_name: str,
                          sample_size: int = 100) -> Optional[Dict[str, Any]]:
        """
        Gera explicações SHAP para o modelo (se disponível).

        Args:
            X: Features
            model_name: Nome do modelo
            sample_size: Tamanho da amostra para SHAP

        Returns:
            Dicionário com explicações SHAP ou None se indisponível
        """
        if not SHAP_AVAILABLE:
            print("⚠️ SHAP não disponível para explicações")
            return None

        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")

        pipeline = self.trained_models[model_name]

        # Amostra para explicação
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)

        # Remover scaler para SHAP (explicar features originais)
        model = pipeline.named_steps['model']

        try:
            if isinstance(model, (XGBClassifier, LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, RandomForestClassifier):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)

            shap_values = explainer(X_sample)

            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_sample': X_sample
            }

        except Exception as e:
            print(f"⚠️ Erro ao gerar explicações SHAP: {e}")
            return None

    def select_best_aml_model(self, model_comparison_df: pd.DataFrame, 
                             all_evaluation_results: List[Dict]) -> Tuple[str, Dict, str]:
        """
        Seleciona o melhor modelo para AML baseado em critérios específicos do domínio.
        Versão simplificada e robusta.
        """
        print("🎯 Selecionando Melhor Modelo AML...")
        print("=" * 50)

        # Obter nomes dos modelos da comparação
        model_names = [name.lower() for name in model_comparison_df['model'].str.lower().tolist()]

        # Filtrar apenas modelos compliant
        compliant_models = model_comparison_df[model_comparison_df['compliant'] == True]

        if len(compliant_models) == 0:
            print("⚠️ Nenhum modelo atende aos requisitos regulatórios!")
            print("   Selecionando modelo com melhor recall...")

            # Fallback: melhor recall
            best_model_row = model_comparison_df.loc[model_comparison_df['recall_test'].idxmax()]
            best_model_name = best_model_row['model'].lower()
            justification = "fallback_recall"

        elif len(compliant_models) == 1:
            best_model_row = compliant_models.iloc[0]
            best_model_name = best_model_row['model'].lower()
            justification = "unique_compliant"

        else:
            print(f"✅ {len(compliant_models)} modelos compliant encontrados")
            print("   Selecionando o melhor baseado em recall...")

            # Simples: escolher o com melhor recall entre compliant
            best_model_row = compliant_models.loc[compliant_models['recall_test'].idxmax()]
            best_model_name = best_model_row['model'].lower()
            justification = "best_recall_among_compliant"

        # Obter métricas básicas do melhor modelo
        optimal_metrics = {
            'recall': best_model_row.get('recall_test', 0.0),
            'precision': best_model_row.get('precision_test', 0.0),
            'f1': best_model_row.get('f1_test', 0.0)
        }

        # Justificativa detalhada
        justification_map = {
            'unique_compliant': "Único modelo que atende requisitos regulatórios",
            'best_recall_among_compliant': "Melhor recall entre modelos compliant",
            'fallback_recall': f"Melhor recall disponível ({optimal_metrics['recall']:.3f}) - revisar compliance"
        }

        print(f"\n🏆 Melhor Modelo Selecionado: {best_model_name.upper()}")
        print(f"   Justificativa: {justification_map.get(justification, justification)}")
        print(f"   Recall: {optimal_metrics['recall']:.3f}")
        print(f"   Precision: {optimal_metrics['precision']:.3f}")
        print(f"   F1-Score: {optimal_metrics['f1']:.3f}")
        print(f"   ROC-AUC CV: {best_model_row['roc_auc_cv']:.4f}")
        print(f"   Compliant: {'✅ Sim' if best_model_row['compliant'] else '❌ Não'}")

        return best_model_name, optimal_metrics, justification

    def get_best_model(self) -> Tuple[str, Any]:
        """
        Retorna o melhor modelo baseado em AUC da validação cruzada.
        DEPRECATED: Use select_best_aml_model para seleção adequada em AML.

        Returns:
            Tupla (nome_modelo, pipeline)
        """
        print("⚠️ AVISO: get_best_model() usa apenas ROC-AUC.")
        print("   Para AML, considere usar select_best_aml_model() com critérios específicos do domínio.")

        if not self.cv_results:
            raise ValueError("Nenhum modelo foi treinado ainda")

        best_model = max(self.cv_results.items(),
                        key=lambda x: x[1]['roc_auc']['mean'])

        return best_model[0], self.trained_models[best_model[0]]

    def save_training_metadata(self, model_name: str, training_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any], save_path: str) -> None:
        """
        Salva metadados completos do treinamento em JSON.

        Args:
            model_name: Nome do modelo
            training_results: Resultados do treinamento
            evaluation_results: Resultados da avaliação
            save_path: Caminho para salvar o arquivo JSON
        """
        import json
        from datetime import datetime

        # Coletar parâmetros do modelo
        model_config = self.config['models'][model_name]
        pipeline = training_results['pipeline']
        model = pipeline.named_steps['model']

        # Informações do modelo
        model_info = {
            'model_name': model_name,
            'model_type': model_config['model_type'],
            'training_timestamp': datetime.now().isoformat(),
            'parameters': model_config['params'],
            'actual_parameters': {
                'n_estimators': getattr(model, 'n_estimators', None),
                'max_depth': getattr(model, 'max_depth', None),
                'learning_rate': getattr(model, 'learning_rate', None),
                'subsample': getattr(model, 'subsample', None),
                'colsample_bytree': getattr(model, 'colsample_bytree', None),
                'scale_pos_weight': getattr(model, 'scale_pos_weight', None),
                'class_weight': getattr(model, 'class_weight', None),
            }
        }

        # Informações de early stopping
        early_stopping_info = {}
        if hasattr(model, 'best_iteration'):
            early_stopping_info = {
                'early_stopping_used': True,
                'best_iteration': model.best_iteration,
                'max_iterations': model_config['params']['n_estimators'],
                'efficiency_ratio': model.best_iteration / model_config['params']['n_estimators']
            }
        elif hasattr(model, 'best_iteration_'):  # LightGBM
            early_stopping_info = {
                'early_stopping_used': True,
                'best_iteration': model.best_iteration_,
                'max_iterations': model_config['params']['n_estimators'],
                'efficiency_ratio': model.best_iteration_ / model_config['params']['n_estimators']
            }
        else:
            early_stopping_info = {
                'early_stopping_used': False,
                'fixed_estimators': getattr(model, 'n_estimators', 'N/A')
            }

        # Métricas de validação cruzada
        cv_info = {
            'temporal_splits': self.config['temporal_splits'],
            'metrics': {}
        }

        for metric_name, metric_data in training_results['cv_results'].items():
            cv_info['metrics'][metric_name] = {
                'mean': metric_data['mean'],
                'std': metric_data['std'],
                'scores': metric_data['scores'].tolist()
            }

        # Métricas de avaliação final
        evaluation_info = {
            'roc_auc': evaluation_results['roc_auc'],
            'pr_auc': evaluation_results['pr_auc'],
            'optimal_threshold': evaluation_results['optimal_threshold'],
            'threshold_analysis': evaluation_results['threshold_analysis'].to_dict('records'),
            'cost_analysis': evaluation_results['cost_analysis'],
            'regulatory_compliance': evaluation_results['regulatory_compliance']
        }

        # Configuração experimental
        config_info = {
            'random_seed': self.config['random_seed'],
            'early_stopping_config': self.config['early_stopping'],
            'business_metrics': self.config['business_metrics'],
            'aml_thresholds': self.config['aml_thresholds']
        }

        # Metadata completo
        metadata = {
            'model_info': model_info,
            'early_stopping': early_stopping_info,
            'cross_validation': cv_info,
            'evaluation': evaluation_info,
            'configuration': config_info,
            'training_duration_seconds': training_results.get('training_time_seconds', None)
        }

        # Salvar em JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"💾 Metadata salva: {save_path}")

    def save_all_models(self, models_dict: Dict[str, Any], base_path: str) -> Dict[str, str]:
        """
        Salva todos os modelos treinados em arquivos PKL.

        Args:
            models_dict: Dicionário com modelos treinados
            base_path: Caminho base para salvar

        Returns:
            Dicionário com caminhos dos arquivos salvos
        """
        import joblib
        from pathlib import Path

        saved_paths = {}
        base_path = Path(base_path)

        for model_name, pipeline in models_dict.items():
            # Caminho do arquivo
            model_path = base_path / f'aml_model_{model_name}.pkl'

            # Metadata completa do modelo
            model_metadata = {
                'model': pipeline,
                'model_name': model_name,
                'training_config': self.config,
                'training_timestamp': pd.Timestamp.now(),
                'model_version': '1.0.0'
            }

            # Salvar
            joblib.dump(model_metadata, model_path)
            saved_paths[model_name] = str(model_path)

            print(f"💾 Modelo salvo: {model_path}")

        return saved_paths

    def check_model_cache(self, model_name: str, artifacts_dir: str) -> bool:
        """
        Verifica se modelo e metadata existem no cache.
        
        Args:
            model_name: Nome do modelo
            artifacts_dir: Diretório de artefatos
            
        Returns:
            True se ambos existem
        """
        from pathlib import Path
        models_dir = Path(artifacts_dir) / 'trained_models'
        model_path = models_dir / f'aml_model_{model_name}.pkl'
        metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'
        return model_path.exists() and metadata_path.exists()

    def load_model_from_cache(self, model_name: str, artifacts_dir: str) -> Tuple[Any, Dict, Dict]:
        """
        Carrega modelo, training_results e evaluation_results do cache.
        
        Args:
            model_name: Nome do modelo
            artifacts_dir: Diretório de artefatos
            
        Returns:
            modelo, training_results, evaluation_results
        """
        import joblib
        import json
        from pathlib import Path
        
        models_dir = Path(artifacts_dir) / 'trained_models'
        model_path = models_dir / f'aml_model_{model_name}.pkl'
        metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'
        
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        training_results = metadata.get('training_results', {})
        evaluation_results = metadata.get('evaluation_results', {})
        
        return model, training_results, evaluation_results

    def save_model_to_cache(self, model_name: str, model: Any, training_results: Dict, evaluation_results: Dict, artifacts_dir: str):
        """
        Salva modelo e metadata no cache.

        Args:
            model_name: Nome do modelo
            model: Modelo treinado
            training_results: Resultados do treinamento
            evaluation_results: Resultados da avaliação
            artifacts_dir: Diretório de artefatos
        """
        import joblib
        import json
        from pathlib import Path

        models_dir = Path(artifacts_dir) / 'trained_models'
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f'aml_model_{model_name}.pkl'
        metadata_path = models_dir / f'aml_training_metadata_{model_name}.json'

        joblib.dump(model, model_path)

        # Função auxiliar para tornar objetos serializáveis
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            else:
                # Para outros objetos, tentar converter para string ou retornar None
                try:
                    return str(obj)
                except:
                    return None

        # Tornar resultados serializáveis
        serializable_training = make_serializable(training_results)
        serializable_evaluation = make_serializable(evaluation_results)

        metadata = {
            'training_results': serializable_training,
            'evaluation_results': serializable_evaluation,
            'timestamp': str(datetime.now())
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
def create_aml_experiment_config() -> Dict[str, Any]:
    """
    Cria configuração experimental padrão para AML com early stopping.

    Returns:
        Dicionário com configuração experimental
    """
    return {
        'random_seed': 42,
        'temporal_splits': 5,
        'early_stopping': {
            'enabled': True,
            'rounds': 20,  # Rounds sem melhoria para parar
            'metric': 'auc',  # Métrica para monitorar (para XGBoost/LightGBM)
            'min_delta': 0.001,  # Melhoria mínima para considerar
            'max_rounds': 1000  # Máximo de rounds independentemente
        },
        'models': {
            'xgboost': {
                'model_type': 'xgb',
                'params': {
                    'n_estimators': 1000,  # Aumentado para early stopping
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'scale_pos_weight': 10,
                    'eval_metric': 'auc',  # Para early stopping
                    'use_label_encoder': False,
                    'verbosity': 0
                }
            },
            'lightgbm': {
                'model_type': 'lgb',
                'params': {
                    'n_estimators': 1000,  # Aumentado para early stopping
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'scale_pos_weight': 10,
                    'verbosity': -1,
                    'metric': 'auc'  # Para early stopping
                }
            },
            'random_forest': {
                'model_type': 'rf',
                'params': {
                    'n_estimators': 100,  # RF não usa early stopping tradicional
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                }
            }
        },
        'metrics': ['roc_auc', 'average_precision', 'recall', 'precision', 'f1'],
        'aml_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9],
        'business_metrics': {
            'cost_benefit_ratio': {'fp_cost': 1, 'fn_cost': 100},
            'regulatory_requirements': {
                'min_recall': 0.8,
                'max_false_positive_rate': 0.05
            }
        }
    }


def load_aml_model(model_path: str) -> Dict[str, Any]:
    """
    Carrega modelo AML salvo com metadados.

    Args:
        model_path: Caminho para o arquivo do modelo

    Returns:
        Dicionário com modelo e metadados
    """
    import joblib

    model_data = joblib.load(model_path)

    required_keys = ['model', 'threshold', 'metrics']
    for key in required_keys:
        if key not in model_data:
            raise ValueError(f"Arquivo de modelo incompleto: falta chave '{key}'")

    return model_data


def predict_aml_transaction(model_data: Dict[str, Any], features: np.ndarray) -> Dict[str, Any]:
    """
    Faz predição AML para uma transação.

    Args:
        model_data: Dados do modelo carregado
        features: Features da transação

    Returns:
        Dicionário com predição e explicações
    """
    model = model_data['model']
    threshold = model_data['threshold']

    # Predição
    pred_proba = model.predict_proba(features.reshape(1, -1))[0, 1]
    pred_class = 1 if pred_proba >= threshold else 0

    return {
        'prediction': pred_class,
        'probability': pred_proba,
        'threshold': threshold,
        'status': 'SUSPEITA' if pred_class == 1 else 'APROVADA',
        'confidence': pred_proba if pred_class == 1 else (1 - pred_proba)
    }

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import recall_score, precision_score
import numpy as np

class AMLTrainingCallback:
    """
    Callback customizado para monitorar métricas AML durante treinamento.
    Permite early stopping baseado em critérios específicos do domínio.
    """

    def __init__(self, X_val, y_val, config, patience=10, min_delta=0.001):
        self.X_val = X_val
        self.y_val = y_val
        self.config = config
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.history = []

        # Critérios AML
        self.regulatory_recall_min = config['business_metrics']['regulatory_requirements']['min_recall']
        self.regulatory_fp_max = config['business_metrics']['regulatory_requirements']['max_false_positive_rate']
        self.cost_weights = config['business_metrics']['cost_benefit_ratio']

    def __call__(self, env):
        """Chamado a cada iteração/epoch do treinamento."""
        # Obter predições atuais
        current_model = env.model
        y_pred_proba = current_model.predict(self.X_val)

        # Calcular métricas AML
        metrics = self._calculate_aml_metrics(y_pred_proba)

        # Armazenar no histórico
        self.history.append({
            'iteration': env.iteration,
            'metrics': metrics
        })

        # Verificar early stopping
        current_score = self._calculate_aml_score(metrics)

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            # Salvar melhores pesos (se suportado)
            if hasattr(current_model, 'save_model'):
                self.best_weights = current_model.copy()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = env.iteration
            print(f"\n🛑 Early stopping at iteration {env.iteration} (patience={self.patience})")
            print(f"   Best AML score: {self.best_score:.4f}")
            raise StopIteration()

    def _calculate_aml_metrics(self, y_pred_proba):
        """Calcula métricas específicas para AML."""
        metrics = {}

        # Testar diferentes thresholds
        for threshold in self.config['aml_thresholds']:
            y_pred = (y_pred_proba >= threshold).astype(int)

            recall = recall_score(self.y_val, y_pred, zero_division=0)
            precision = precision_score(self.y_val, y_pred, zero_division=0)
            fraud_rate = y_pred.mean()

            # Cálculo de custo
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            total_cost = (fp * self.cost_weights['fp_cost']) + (fn * self.cost_weights['fn_cost'])
            cost_per_transaction = total_cost / len(self.y_val)

            metrics[f'threshold_{threshold}'] = {
                'recall': recall,
                'precision': precision,
                'fraud_rate': fraud_rate,
                'total_cost': total_cost,
                'cost_per_transaction': cost_per_transaction,
                'compliant': (recall >= self.regulatory_recall_min and
                            fraud_rate <= self.regulatory_fp_max)
            }

        return metrics

    def _calculate_aml_score(self, metrics):
        """
        Calcula score AML composto baseado em múltiplos fatores.
        Prioriza: compliance > recall > custo-benefício > precision
        """
        # Encontrar melhor threshold
        best_threshold_data = None
        best_score = -np.inf

        for threshold_key, threshold_data in metrics.items():
            # Score composto
            compliance_score = 1000 if threshold_data['compliant'] else 0
            recall_score = threshold_data['recall'] * 100
            cost_score = max(0, 1 - threshold_data['cost_per_transaction']) * 50  # Normalizar custo
            precision_score = threshold_data['precision'] * 25

            total_score = compliance_score + recall_score + cost_score + precision_score

            if total_score > best_score:
                best_score = total_score
                best_threshold_data = threshold_data

        return best_score

    def get_best_metrics(self):
        """Retorna as melhores métricas encontradas."""
        if not self.history:
            return None

        best_entry = max(self.history, key=lambda x: self._calculate_aml_score(x['metrics']))
        return best_entry

    def get_training_summary(self):
        """Retorna resumo do treinamento."""
        if not self.history:
            return {}

        return {
            'total_iterations': len(self.history),
            'stopped_early': self.stopped_epoch > 0,
            'stopped_at_iteration': self.stopped_epoch,
            'best_iteration': self.get_best_metrics()['iteration'] if self.get_best_metrics() else None,
            'best_score': self.best_score,
            'history': self.history
        }

class StreamlitLiveCallback:
    """
    Callback para plotar métricas de treinamento em tempo real no Streamlit.
    """

    def __init__(self, model_name, update_interval=5, placeholder=None, st_module=None):
        self.model_name = model_name
        self.update_interval = update_interval
        self.placeholder = placeholder
        self.st = st_module  # Módulo streamlit passado como parâmetro
        self.iterations = []
        self.train_auc = []
        self.val_auc = []
        self.oob_scores = []
        self.start_time = time.time()

    def add_metrics(self, iteration, train_auc=None, val_auc=None, oob_score=None):
        """Adiciona métricas e atualiza dashboard se necessário."""
        self.iterations.append(iteration)
        if train_auc is not None:
            self.train_auc.append(train_auc)
        if val_auc is not None:
            self.val_auc.append(val_auc)
        if oob_score is not None:
            self.oob_scores.append(oob_score)

        if iteration % self.update_interval == 0 or iteration == 1:
            self.update_dashboard()

    def update_dashboard(self):
        """Atualiza o dashboard Streamlit em tempo real."""
        if self.placeholder is None or self.st is None:
            return

        with self.placeholder.container():
            # KPIs
            col1, col2, col3, col4 = self.st.columns(4)

            with col1:
                current_iter = self.iterations[-1] if self.iterations else 0
                total_iters = getattr(self, 'total_iterations', 100)  # Será definido depois
                progress = current_iter / total_iters if total_iters > 0 else 0
                self.st.metric(
                    label="Iteração Atual 🔄",
                    value=f"{current_iter}/{total_iters}",
                    delta=f"{progress:.1%} concluído"
                )

            with col2:
                current_auc = self.val_auc[-1] if self.val_auc else 0
                prev_auc = self.val_auc[-2] if len(self.val_auc) > 1 else 0
                delta_auc = current_auc - prev_auc if prev_auc > 0 else current_auc
                self.st.metric(
                    label="AUC Validação 📈",
                    value=f"{current_auc:.4f}",
                    delta=f"{delta_auc:+.4f}"
                )

            with col3:
                elapsed = time.time() - self.start_time
                self.st.metric(
                    label="Tempo Decorrido ⏱️",
                    value=f"{elapsed:.1f}s"
                )

            with col4:
                if self.oob_scores:
                    current_oob = self.oob_scores[-1]
                    prev_oob = self.oob_scores[-2] if len(self.oob_scores) > 1 else 0
                    delta_oob = current_oob - prev_oob if prev_oob > 0 else current_oob
                    self.st.metric(
                        label="OOB Score 🌳",
                        value=f"{current_oob:.4f}",
                        delta=f"{delta_oob:+.4f}"
                    )
                else:
                    eta = self._estimate_eta()
                    self.st.metric(
                        label="ETA ⏳",
                        value=f"{eta:.1f}s"
                    )

            # Gráfico de progresso
            self.st.subheader("📊 Curvas de Aprendizado em Tempo Real")

            fig = go.Figure()

            # AUC de validação
            if self.val_auc:
                fig.add_trace(go.Scatter(
                    x=self.iterations[:len(self.val_auc)],
                    y=self.val_auc,
                    mode='lines+markers',
                    name='AUC Validação',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6)
                ))

            # AUC de treino
            if self.train_auc:
                fig.add_trace(go.Scatter(
                    x=self.iterations[:len(self.train_auc)],
                    y=self.train_auc,
                    mode='lines+markers',
                    name='AUC Treino',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=4)
                ))

            # OOB Score se disponível
            if self.oob_scores:
                fig.add_trace(go.Scatter(
                    x=self.iterations[:len(self.oob_scores)],
                    y=self.oob_scores,
                    mode='lines+markers',
                    name='OOB Score',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    yaxis='y2'
                ))

            fig.update_layout(
                title=f"{self.model_name.upper()} - Treinamento em Tempo Real",
                xaxis_title="Iterações",
                yaxis_title="AUC",
                yaxis2=dict(
                    title="OOB Score",
                    overlaying="y",
                    side="right"
                ),
                height=400,
                showlegend=True,
                hovermode='x unified'
            )

            self.st.plotly_chart(fig, use_container_width=True)

            # Barra de progresso
            progress_bar = self.st.progress(progress)
            self.st.text(f"Treinamento: {progress:.1%} concluído")

    def _estimate_eta(self):
        """Estima tempo restante."""
        if len(self.iterations) < 2:
            return 0

        elapsed = time.time() - self.start_time
        current_iter = self.iterations[-1]
        total_iters = getattr(self, 'total_iterations', 100)

        if current_iter >= total_iters:
            return 0

        avg_time_per_iter = elapsed / current_iter
        remaining_iters = total_iters - current_iter
        return avg_time_per_iter * remaining_iters

    def finalize_plot(self):
        """Finaliza o plot com métricas finais."""
        if self.placeholder and self.st:
            with self.placeholder.container():
                self.st.success("🎉 Treinamento Concluído!")

                # Resumo final
                if self.val_auc:
                    final_auc = self.val_auc[-1]
                    self.st.metric("AUC Final", f"{final_auc:.4f}")

                if self.oob_scores:
                    final_oob = self.oob_scores[-1]
                    self.st.metric("OOB Score Final", f"{final_oob:.4f}")

                elapsed = time.time() - self.start_time
                self.st.metric("Tempo Total", f"{elapsed:.1f}s")