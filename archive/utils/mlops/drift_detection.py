"""
Drift Detection para Monitoramento em Produção
===============================================

Detecta dois tipos de drift:
1. **Data Drift**: Distribuição de features muda (P(X) muda)
2. **Concept Drift**: Relação X→Y muda (P(Y|X) muda)

Testes Estatísticos:
--------------------
- **PSI (Population Stability Index)**: Features numéricas
- **KS (Kolmogorov-Smirnov)**: Distribuições contínuas
- **Chi-Square**: Features categóricas
- **JS Divergence (Jensen-Shannon)**: Divergência entre distribuições

Thresholds:
-----------
PSI:
- < 0.1: Sem drift
- 0.1 - 0.25: Drift moderado (monitorar)
- > 0.25: Drift severo (retreinar!)

KS Statistic:
- p-value > 0.05: Sem drift
- p-value < 0.05: Drift detectado

Autor: Time de Data Science
Data: Outubro 2025
Fase: 4.2 - Monitoramento em Produção
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Resultado de detecção de drift para uma feature."""
    feature: str
    drift_score: float
    drift_detected: bool
    drift_severity: str  # 'No Drift', 'Moderate', 'Severe'
    test_statistic: float
    p_value: Optional[float]
    test_type: str  # 'PSI', 'KS', 'Chi-Square', 'JS'
    reference_distribution: Optional[np.ndarray] = None
    current_distribution: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        d = asdict(self)
        # Remover arrays grandes
        d.pop('reference_distribution', None)
        d.pop('current_distribution', None)
        return d


class DriftDetector:
    """
    Detector de drift em features.
    
    Parâmetros
    ----------
    psi_threshold_moderate : float, default=0.1
        Threshold para drift moderado (PSI)
    psi_threshold_severe : float, default=0.25
        Threshold para drift severo (PSI)
    ks_p_value_threshold : float, default=0.05
        P-value threshold para KS test
    n_bins : int, default=10
        Número de bins para PSI
        
    Exemplo
    -------
    >>> detector = DriftDetector()
    >>> 
    >>> # Treinar com dados de referência
    >>> detector.fit(X_reference)
    >>> 
    >>> # Detectar drift em dados novos
    >>> drift_report = detector.detect(X_current)
    >>> detector.plot_drift_report(drift_report)
    """
    
    def __init__(
        self,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_severe: float = 0.25,
        ks_p_value_threshold: float = 0.05,
        n_bins: int = 10
    ):
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_severe = psi_threshold_severe
        self.ks_p_value_threshold = ks_p_value_threshold
        self.n_bins = n_bins
        
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Any] = {}
        
        logger.info("🔍 DriftDetector inicializado")
    
    def fit(self, X_reference: pd.DataFrame):
        """
        Treina detector com dados de referência (baseline).
        
        Parâmetros
        ----------
        X_reference : pd.DataFrame
            Dados de referência (treino)
        """
        self.reference_data = X_reference.copy()
        
        # Computar estatísticas de referência
        for col in X_reference.columns:
            if pd.api.types.is_numeric_dtype(X_reference[col]):
                self.reference_stats[col] = {
                    'type': 'numeric',
                    'mean': X_reference[col].mean(),
                    'std': X_reference[col].std(),
                    'min': X_reference[col].min(),
                    'max': X_reference[col].max(),
                    'quartiles': X_reference[col].quantile([0.25, 0.5, 0.75]).tolist()
                }
            else:
                value_counts = X_reference[col].value_counts(normalize=True)
                self.reference_stats[col] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict()
                }
        
        logger.info(f"✅ Referência estabelecida: {len(X_reference)} amostras, {len(X_reference.columns)} features")
    
    def detect(
        self,
        X_current: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> List[DriftResult]:
        """
        Detecta drift entre referência e dados atuais.
        
        Parâmetros
        ----------
        X_current : pd.DataFrame
            Dados atuais
        features : List[str], optional
            Features específicas (default: todas)
            
        Returns
        -------
        drift_results : List[DriftResult]
            Lista de resultados de drift
        """
        if self.reference_data is None:
            raise ValueError("Detector não foi treinado. Execute .fit() primeiro.")
        
        if features is None:
            features = self.reference_data.columns.tolist()
        
        results = []
        
        for feature in features:
            if feature not in self.reference_data.columns:
                logger.warning(f"⚠️ Feature '{feature}' não existe em referência")
                continue
            
            if feature not in X_current.columns:
                logger.warning(f"⚠️ Feature '{feature}' não existe em dados atuais")
                continue
            
            # Detectar tipo e aplicar teste apropriado
            if self.reference_stats[feature]['type'] == 'numeric':
                result = self._detect_numeric(feature, X_current[feature])
            else:
                result = self._detect_categorical(feature, X_current[feature])
            
            results.append(result)
        
        # Estatísticas
        n_drift = sum(r.drift_detected for r in results)
        n_severe = sum(r.drift_severity == 'Severe' for r in results)
        
        logger.info(f"🔍 Drift Detection completa:")
        logger.info(f"   Features analisadas: {len(results)}")
        logger.info(f"   Features com drift: {n_drift} ({n_drift/len(results)*100:.1f}%)")
        logger.info(f"   Features com drift severo: {n_severe}")
        
        return results
    
    def _detect_numeric(self, feature: str, current_data: pd.Series) -> DriftResult:
        """Detecta drift em feature numérica (PSI + KS test)."""
        reference = self.reference_data[feature].dropna()
        current = current_data.dropna()
        
        # 1. PSI (Population Stability Index)
        psi_score = self._calculate_psi(reference, current)
        
        # 2. KS Test (Kolmogorov-Smirnov)
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        
        # Classificar severidade (usar PSI como métrica principal)
        if psi_score < self.psi_threshold_moderate:
            severity = 'No Drift'
            detected = False
        elif psi_score < self.psi_threshold_severe:
            severity = 'Moderate'
            detected = True
        else:
            severity = 'Severe'
            detected = True
        
        return DriftResult(
            feature=feature,
            drift_score=psi_score,
            drift_detected=detected,
            drift_severity=severity,
            test_statistic=ks_stat,
            p_value=ks_pvalue,
            test_type='PSI+KS',
            reference_distribution=reference.values,
            current_distribution=current.values
        )
    
    def _detect_categorical(self, feature: str, current_data: pd.Series) -> DriftResult:
        """Detecta drift em feature categórica (Chi-Square + JS divergence)."""
        reference = self.reference_data[feature].dropna()
        current = current_data.dropna()
        
        # Distribuições
        ref_dist = reference.value_counts(normalize=True).sort_index()
        cur_dist = current.value_counts(normalize=True).sort_index()
        
        # Alinhar categorias
        all_categories = sorted(set(ref_dist.index) | set(cur_dist.index))
        ref_dist = ref_dist.reindex(all_categories, fill_value=1e-10)
        cur_dist = cur_dist.reindex(all_categories, fill_value=1e-10)
        
        # 1. Chi-Square Test
        # Converter para contagens
        ref_counts = (ref_dist * len(reference)).astype(int)
        cur_counts = (cur_dist * len(current)).astype(int)
        
        chi2_stat, chi2_pvalue = stats.chisquare(cur_counts, ref_counts)
        
        # 2. Jensen-Shannon Divergence
        js_div = jensenshannon(ref_dist.values, cur_dist.values)
        
        # Classificar severidade (usar JS divergence)
        # JS divergence range: [0, 1], onde 0 = idêntico, 1 = totalmente diferente
        if js_div < 0.1:
            severity = 'No Drift'
            detected = False
        elif js_div < 0.25:
            severity = 'Moderate'
            detected = True
        else:
            severity = 'Severe'
            detected = True
        
        return DriftResult(
            feature=feature,
            drift_score=js_div,
            drift_detected=detected,
            drift_severity=severity,
            test_statistic=chi2_stat,
            p_value=chi2_pvalue,
            test_type='Chi-Square+JS',
            reference_distribution=ref_dist.values,
            current_distribution=cur_dist.values
        )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Calcula Population Stability Index (PSI).
        
        PSI = Σ (current% - reference%) * ln(current% / reference%)
        
        Interpretação:
        - < 0.1: Sem drift
        - 0.1 - 0.25: Drift moderado
        - > 0.25: Drift severo
        """
        # Criar bins baseados na referência
        _, bins = pd.cut(reference, bins=self.n_bins, retbins=True, duplicates='drop')
        
        # Binning
        ref_binned = pd.cut(reference, bins=bins, include_lowest=True)
        cur_binned = pd.cut(current, bins=bins, include_lowest=True)
        
        # Distribuições
        ref_dist = ref_binned.value_counts(normalize=True).sort_index()
        cur_dist = cur_binned.value_counts(normalize=True).sort_index()
        
        # Alinhar
        ref_dist = ref_dist.reindex(cur_dist.index, fill_value=1e-10)
        
        # Evitar divisão por zero
        ref_dist = ref_dist.replace(0, 1e-10)
        cur_dist = cur_dist.replace(0, 1e-10)
        
        # PSI
        psi = ((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum()
        
        return psi
    
    def get_drift_summary(self, drift_results: List[DriftResult]) -> pd.DataFrame:
        """
        Retorna DataFrame com resumo de drift.
        
        Parâmetros
        ----------
        drift_results : List[DriftResult]
            Resultados de drift
            
        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame com resumo
        """
        df = pd.DataFrame([r.to_dict() for r in drift_results])
        df = df.sort_values('drift_score', ascending=False)
        
        return df
    
    def plot_drift_report(
        self,
        drift_results: List[DriftResult],
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plota relatório de drift.
        
        Parâmetros
        ----------
        drift_results : List[DriftResult]
            Resultados de drift
        top_n : int
            Top N features com maior drift
        save_path : str, optional
            Caminho para salvar figura
        """
        # Ordenar por drift score
        results_sorted = sorted(drift_results, key=lambda x: x.drift_score, reverse=True)
        top_results = results_sorted[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Drift Score (Top N)
        ax = axes[0, 0]
        features = [r.feature for r in top_results]
        scores = [r.drift_score for r in top_results]
        colors = ['red' if r.drift_severity == 'Severe' else 'orange' if r.drift_severity == 'Moderate' else 'green' 
                  for r in top_results]
        
        ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Drift Score', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Drift Score', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Severe'),
            Patch(facecolor='orange', alpha=0.7, label='Moderate'),
            Patch(facecolor='green', alpha=0.7, label='No Drift')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # 2. Distribuição de Severidade
        ax = axes[0, 1]
        severity_counts = pd.Series([r.drift_severity for r in drift_results]).value_counts()
        colors_pie = {'Severe': 'red', 'Moderate': 'orange', 'No Drift': 'green'}
        colors_list = [colors_pie.get(s, 'gray') for s in severity_counts.index]
        
        ax.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
               colors=colors_list, startangle=90)
        ax.set_title('Drift Severity Distribution', fontsize=14, fontweight='bold')
        
        # 3. Distribuição de feature específica (maior drift)
        if top_results:
            ax = axes[1, 0]
            worst_drift = top_results[0]
            
            if worst_drift.reference_distribution is not None and worst_drift.current_distribution is not None:
                ax.hist(worst_drift.reference_distribution, bins=30, alpha=0.6, label='Reference', color='blue', density=True)
                ax.hist(worst_drift.current_distribution, bins=30, alpha=0.6, label='Current', color='red', density=True)
                ax.set_xlabel('Value', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title(f'Distribution: {worst_drift.feature} (Worst Drift)', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        # 4. P-values (se disponível)
        ax = axes[1, 1]
        results_with_pval = [r for r in top_results if r.p_value is not None]
        
        if results_with_pval:
            features_pval = [r.feature for r in results_with_pval]
            pvalues = [r.p_value for r in results_with_pval]
            
            ax.barh(range(len(features_pval)), pvalues, color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(features_pval)))
            ax.set_yticklabels(features_pval)
            ax.set_xlabel('P-Value', fontsize=12)
            ax.set_title('Statistical Test P-Values', fontsize=14, fontweight='bold')
            ax.axvline(0.05, color='red', linestyle='--', label='p=0.05 threshold')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No p-values available', ha='center', va='center', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Relatório salvo: {save_path}")
        
        plt.show()
    
    def generate_alert(self, drift_results: List[DriftResult]) -> Dict[str, Any]:
        """
        Gera alerta de drift para sistema de monitoramento.
        
        Parâmetros
        ----------
        drift_results : List[DriftResult]
            Resultados de drift
            
        Returns
        -------
        alert : Dict
            Alerta estruturado
        """
        severe_drifts = [r for r in drift_results if r.drift_severity == 'Severe']
        moderate_drifts = [r for r in drift_results if r.drift_severity == 'Moderate']
        
        alert_level = 'CRITICAL' if severe_drifts else ('WARNING' if moderate_drifts else 'OK')
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert_level,
            'total_features': len(drift_results),
            'features_with_drift': len([r for r in drift_results if r.drift_detected]),
            'severe_drifts': len(severe_drifts),
            'moderate_drifts': len(moderate_drifts),
            'top_drifting_features': [
                {'feature': r.feature, 'score': r.drift_score, 'severity': r.drift_severity}
                for r in sorted(drift_results, key=lambda x: x.drift_score, reverse=True)[:5]
            ],
            'recommendation': self._get_recommendation(severe_drifts, moderate_drifts)
        }
        
        return alert
    
    def _get_recommendation(self, severe_drifts: List, moderate_drifts: List) -> str:
        """Gera recomendação baseada em drift."""
        if severe_drifts:
            return "🚨 AÇÃO IMEDIATA: Drift severo detectado. Retreinar modelo URGENTE ou pausar predições."
        elif moderate_drifts:
            return "⚠️ MONITORAR: Drift moderado detectado. Agendar retreinamento em breve."
        else:
            return "✅ OK: Nenhum drift significativo. Modelo estável."


# Exemplo de uso
if __name__ == "__main__":
    print("="*80)
    print("TESTE: Drift Detection")
    print("="*80)
    
    # Simular dados
    np.random.seed(42)
    
    # Dados de referência (treino)
    n_ref = 5000
    X_reference = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_ref),
        'feature_2': np.random.exponential(2, n_ref),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_ref, p=[0.5, 0.3, 0.2]),
        'feature_4': np.random.uniform(-5, 5, n_ref),
        'feature_5': np.random.normal(10, 2, n_ref)
    })
    
    # Dados atuais (produção) COM DRIFT
    n_cur = 2000
    X_current = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_cur),  # SEM drift
        'feature_2': np.random.exponential(4, n_cur),  # COM DRIFT (mudou de 2 para 4)
        'feature_3': np.random.choice(['A', 'B', 'C'], n_cur, p=[0.3, 0.4, 0.3]),  # COM DRIFT (mudou distribuição)
        'feature_4': np.random.uniform(-5, 5, n_cur),  # SEM drift
        'feature_5': np.random.normal(15, 2, n_cur)  # COM DRIFT SEVERO (mudou média de 10 para 15)
    })
    
    # Detector
    print("\n1. Treinando detector...")
    detector = DriftDetector()
    detector.fit(X_reference)
    
    # Detecção
    print("\n2. Detectando drift...")
    drift_results = detector.detect(X_current)
    
    # Resumo
    print("\n3. Resumo de Drift:")
    summary = detector.get_drift_summary(drift_results)
    print(summary[['feature', 'drift_score', 'drift_severity', 'test_type']].to_string(index=False))
    
    # Alerta
    print("\n4. Alerta:")
    alert = detector.generate_alert(drift_results)
    print(f"   Level: {alert['alert_level']}")
    print(f"   Recommendation: {alert['recommendation']}")
    print(f"   Top drifting features:")
    for feat in alert['top_drifting_features']:
        print(f"      - {feat['feature']}: {feat['score']:.4f} ({feat['severity']})")
    
    # Plot
    detector.plot_drift_report(drift_results, save_path='drift_report_test.png')
    
    print("\n✅ Teste concluído!")
    print("   Esperado: feature_2, feature_3, feature_5 com drift")
