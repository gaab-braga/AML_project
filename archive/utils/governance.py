"""
Fairness Metrics para Modelos de ML
===================================

Avalia viés e justiça (fairness) em modelos de Machine Learning,
especialmente importante para sistemas AML que impactam decisões financeiras.

Métricas implementadas:
- Demographic Parity
- Equal Opportunity
- Equalized Odds
- Disparate Impact
- Statistical Parity Difference

Autor: Time de Data Science
Data: Outubro 2025
Fase: 3.2 - Fairness e Governança
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


__all__ = [
    'FairnessReport',      # Dataclass para relatório de fairness
    'FairnessAnalyzer',    # Classe principal para análise de fairness
]


@dataclass
class FairnessReport:
    """Container para relatório de fairness."""
    metric_name: str
    overall_value: float
    group_values: Dict[str, float]
    is_fair: bool
    threshold: float
    details: Dict[str, Any]
    
    def __repr__(self):
        status = "✅ FAIR" if self.is_fair else "⚠️ UNFAIR"
        return (f"FairnessReport({self.metric_name}: {status}, "
                f"overall={self.overall_value:.4f}, "
                f"threshold={self.threshold})")


class FairnessAnalyzer:
    """
    Analisa fairness de modelos de ML.
    
    Parâmetros
    ----------
    sensitive_features : List[str]
        Features sensíveis (ex: 'Country', 'Account Type')
    privileged_groups : Dict[str, Any], optional
        Grupos privilegiados por feature (ex: {'Country': 'USA'})
    fairness_threshold : float, default=0.8
        Threshold para disparate impact (0.8 = regra 80%)
        
    Exemplo
    -------
    >>> analyzer = FairnessAnalyzer(
    ...     sensitive_features=['From Bank', 'To Bank'],
    ...     fairness_threshold=0.8
    ... )
    >>> 
    >>> report = analyzer.demographic_parity(
    ...     y_true, y_pred, sensitive_data
    ... )
    >>> analyzer.plot_fairness_comparison([report])
    """
    
    def __init__(
        self,
        sensitive_features: List[str],
        privileged_groups: Optional[Dict[str, Any]] = None,
        fairness_threshold: float = 0.8
    ):
        self.sensitive_features = sensitive_features
        self.privileged_groups = privileged_groups or {}
        self.fairness_threshold = fairness_threshold
        
        logger.info(f"✅ FairnessAnalyzer inicializado")
        logger.info(f"   Features sensíveis: {sensitive_features}")
        logger.info(f"   Threshold: {fairness_threshold} (80% rule)")
    
    def demographic_parity(
        self,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame,
        feature: str
    ) -> FairnessReport:
        """
        Demographic Parity (Statistical Parity).
        
        Mede se diferentes grupos recebem predições positivas na mesma taxa.
        
        P(Y_pred=1 | G=a) ≈ P(Y_pred=1 | G=b)
        
        Parâmetros
        ----------
        y_pred : np.ndarray
            Predições binárias (0/1)
        sensitive_data : pd.DataFrame
            DataFrame com features sensíveis
        feature : str
            Feature sensível para análise
            
        Returns
        -------
        report : FairnessReport
            Relatório de fairness
        """
        logger.info(f"📊 Calculando Demographic Parity para '{feature}'...")
        
        if feature not in sensitive_data.columns:
            raise ValueError(f"Feature '{feature}' não encontrada em sensitive_data")
        
        # Taxa de predições positivas por grupo
        group_values = {}
        for group in sensitive_data[feature].unique():
            mask = sensitive_data[feature] == group
            positive_rate = y_pred[mask].mean()
            group_values[str(group)] = positive_rate
            logger.info(f"   {group}: {positive_rate:.4f} ({positive_rate*100:.2f}%)")
        
        # Calcular disparate impact
        rates = list(group_values.values())
        min_rate = min(rates)
        max_rate = max(rates)
        
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0
        
        # Verificar fairness
        is_fair = disparate_impact >= self.fairness_threshold
        
        logger.info(f"   Disparate Impact: {disparate_impact:.4f}")
        logger.info(f"   Status: {'✅ FAIR' if is_fair else '⚠️ UNFAIR'}")
        
        return FairnessReport(
            metric_name='Demographic Parity',
            overall_value=disparate_impact,
            group_values=group_values,
            is_fair=is_fair,
            threshold=self.fairness_threshold,
            details={
                'min_rate': min_rate,
                'max_rate': max_rate,
                'difference': max_rate - min_rate
            }
        )
    
    def equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame,
        feature: str
    ) -> FairnessReport:
        """
        Equal Opportunity.
        
        Mede se diferentes grupos têm mesma taxa de True Positive (Recall).
        Importante para fraude: queremos detectar fraude independente do grupo.
        
        TPR(G=a) ≈ TPR(G=b)
        
        Parâmetros
        ----------
        y_true : np.ndarray
            Labels verdadeiros
        y_pred : np.ndarray
            Predições
        sensitive_data : pd.DataFrame
            Features sensíveis
        feature : str
            Feature para análise
            
        Returns
        -------
        report : FairnessReport
            Relatório
        """
        logger.info(f"📊 Calculando Equal Opportunity para '{feature}'...")
        
        group_values = {}
        
        for group in sensitive_data[feature].unique():
            mask = sensitive_data[feature] == group
            
            # Calcular TPR (True Positive Rate / Recall)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # TPR = TP / (TP + FN)
            tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
            fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            group_values[str(group)] = tpr
            
            logger.info(f"   {group}: TPR = {tpr:.4f}")
        
        # Disparate impact
        rates = list(group_values.values())
        min_rate = min(rates) if rates else 0
        max_rate = max(rates) if rates else 0
        
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0
        is_fair = disparate_impact >= self.fairness_threshold
        
        logger.info(f"   Disparate Impact: {disparate_impact:.4f}")
        logger.info(f"   Status: {'✅ FAIR' if is_fair else '⚠️ UNFAIR'}")
        
        return FairnessReport(
            metric_name='Equal Opportunity',
            overall_value=disparate_impact,
            group_values=group_values,
            is_fair=is_fair,
            threshold=self.fairness_threshold,
            details={
                'min_tpr': min_rate,
                'max_tpr': max_rate,
                'tpr_difference': max_rate - min_rate
            }
        )
    
    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame,
        feature: str
    ) -> FairnessReport:
        """
        Equalized Odds.
        
        Mede se diferentes grupos têm mesma TPR E FPR.
        Mais rigoroso que Equal Opportunity.
        
        TPR(G=a) ≈ TPR(G=b) E FPR(G=a) ≈ FPR(G=b)
        
        Parâmetros
        ----------
        y_true, y_pred : np.ndarray
            Labels e predições
        sensitive_data : pd.DataFrame
            Features sensíveis
        feature : str
            Feature para análise
            
        Returns
        -------
        report : FairnessReport
            Relatório
        """
        logger.info(f"📊 Calculando Equalized Odds para '{feature}'...")
        
        group_tpr = {}
        group_fpr = {}
        
        for group in sensitive_data[feature].unique():
            mask = sensitive_data[feature] == group
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
            
            # TPR e FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            group_tpr[str(group)] = tpr
            group_fpr[str(group)] = fpr
            
            logger.info(f"   {group}: TPR={tpr:.4f}, FPR={fpr:.4f}")
        
        # Calcular disparate impact para ambos
        tpr_values = list(group_tpr.values())
        fpr_values = list(group_fpr.values())
        
        tpr_di = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0
        fpr_di = min(fpr_values) / max(fpr_values) if max(fpr_values) > 0 else 0
        
        # Overall: pior dos dois
        overall_di = min(tpr_di, fpr_di)
        is_fair = overall_di >= self.fairness_threshold
        
        logger.info(f"   TPR Disparate Impact: {tpr_di:.4f}")
        logger.info(f"   FPR Disparate Impact: {fpr_di:.4f}")
        logger.info(f"   Overall: {overall_di:.4f}")
        logger.info(f"   Status: {'✅ FAIR' if is_fair else '⚠️ UNFAIR'}")
        
        return FairnessReport(
            metric_name='Equalized Odds',
            overall_value=overall_di,
            group_values={'TPR': group_tpr, 'FPR': group_fpr},
            is_fair=is_fair,
            threshold=self.fairness_threshold,
            details={
                'tpr_di': tpr_di,
                'fpr_di': fpr_di,
                'tpr_diff': max(tpr_values) - min(tpr_values),
                'fpr_diff': max(fpr_values) - min(fpr_values)
            }
        )
    
    def statistical_parity_difference(
        self,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame,
        feature: str
    ) -> float:
        """
        Statistical Parity Difference.
        
        Diferença absoluta entre taxas de predição positiva.
        SPD = P(Y_pred=1 | G=a) - P(Y_pred=1 | G=b)
        
        Ideal: próximo de 0
        
        Returns
        -------
        spd : float
            Diferença estatística (ideal: 0)
        """
        group_rates = []
        
        for group in sensitive_data[feature].unique():
            mask = sensitive_data[feature] == group
            positive_rate = y_pred[mask].mean()
            group_rates.append(positive_rate)
        
        spd = max(group_rates) - min(group_rates)
        
        logger.info(f"📊 Statistical Parity Difference: {spd:.4f}")
        
        return spd
    
    def analyze_all_features(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame
    ) -> Dict[str, List[FairnessReport]]:
        """
        Analisa fairness para todas as features sensíveis.
        
        Returns
        -------
        results : Dict[str, List[FairnessReport]]
            Dicionário {feature: [reports]}
        """
        logger.info(f"🔍 Analisando fairness para {len(self.sensitive_features)} features...")
        
        results = {}
        
        for feature in self.sensitive_features:
            if feature not in sensitive_data.columns:
                logger.warning(f"⚠️ Feature '{feature}' não encontrada, pulando...")
                continue
            
            logger.info(f"\n📊 Feature: {feature}")
            logger.info("-" * 60)
            
            feature_reports = []
            
            # Demographic Parity
            try:
                report = self.demographic_parity(y_pred, sensitive_data, feature)
                feature_reports.append(report)
            except Exception as e:
                logger.error(f"Erro em Demographic Parity: {e}")
            
            # Equal Opportunity
            try:
                report = self.equal_opportunity(y_true, y_pred, sensitive_data, feature)
                feature_reports.append(report)
            except Exception as e:
                logger.error(f"Erro em Equal Opportunity: {e}")
            
            # Equalized Odds
            try:
                report = self.equalized_odds(y_true, y_pred, sensitive_data, feature)
                feature_reports.append(report)
            except Exception as e:
                logger.error(f"Erro em Equalized Odds: {e}")
            
            results[feature] = feature_reports
        
        logger.info("\n✅ Análise completa!")
        return results
    
    def plot_fairness_comparison(
        self,
        reports: List[FairnessReport],
        save_path: Optional[str] = None
    ):
        """
        Plot de comparação de métricas de fairness.
        
        Parâmetros
        ----------
        reports : List[FairnessReport]
            Lista de relatórios
        save_path : str, optional
            Caminho para salvar
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Disparate Impact por métrica
        ax = axes[0]
        metrics = [r.metric_name for r in reports]
        values = [r.overall_value for r in reports]
        colors = ['green' if r.is_fair else 'red' for r in reports]
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.7)
        ax.axvline(x=self.fairness_threshold, color='black', linestyle='--', 
                   label=f'Threshold ({self.fairness_threshold})')
        ax.set_xlabel('Disparate Impact', fontsize=12)
        ax.set_title('Fairness Metrics - Disparate Impact', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=10)
        
        # Plot 2: Taxa por grupo (primeiro report)
        if reports:
            ax = axes[1]
            report = reports[0]
            
            if isinstance(report.group_values, dict) and 'TPR' not in report.group_values:
                groups = list(report.group_values.keys())
                values = list(report.group_values.values())
                
                bars = ax.bar(groups, values, alpha=0.7, color='steelblue')
                ax.set_ylabel('Positive Rate', fontsize=12)
                ax.set_xlabel('Group', fontsize=12)
                ax.set_title(f'{report.metric_name} - Group Rates', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Valores
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                           f'{val:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Plot salvo: {save_path}")
        
        plt.show()
    
    def generate_fairness_report_text(
        self,
        results: Dict[str, List[FairnessReport]]
    ) -> str:
        """
        Gera relatório textual de fairness.
        
        Returns
        -------
        report : str
            Relatório em texto
        """
        text = "="*80 + "\n"
        text += "RELATÓRIO DE FAIRNESS\n"
        text += "="*80 + "\n\n"
        
        for feature, reports in results.items():
            text += f"📊 Feature: {feature}\n"
            text += "-"*80 + "\n"
            
            for report in reports:
                status = "✅ FAIR" if report.is_fair else "⚠️ UNFAIR"
                text += f"\n{report.metric_name}: {status}\n"
                text += f"  Disparate Impact: {report.overall_value:.4f} "
                text += f"(threshold: {report.threshold})\n"
                
                if isinstance(report.group_values, dict) and 'TPR' not in report.group_values:
                    text += "  Taxas por grupo:\n"
                    for group, value in report.group_values.items():
                        text += f"    - {group}: {value:.4f}\n"
            
            text += "\n"
        
        # Resumo
        all_reports = [r for reports in results.values() for r in reports]
        total = len(all_reports)
        fair_count = sum(1 for r in all_reports if r.is_fair)
        
        text += "="*80 + "\n"
        text += "RESUMO\n"
        text += "="*80 + "\n"
        text += f"Total de testes: {total}\n"
        text += f"Fair: {fair_count} ({fair_count/total*100:.1f}%)\n"
        text += f"Unfair: {total - fair_count} ({(total-fair_count)/total*100:.1f}%)\n"
        
        if fair_count == total:
            text += "\n✅ Modelo PASSOU em todos os testes de fairness!\n"
        else:
            text += "\n⚠️ Modelo FALHOU em alguns testes. Revisar viés.\n"
        
        return text


# Exemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier
    
    print("="*80)
    print("TESTE: Fairness Analysis")
    print("="*80)
    
    # Gerar dados com feature sensível
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    
    # Adicionar feature sensível (ex: Country)
    np.random.seed(42)
    country = np.random.choice(['USA', 'Brazil', 'India'], size=1000, p=[0.5, 0.3, 0.2])
    
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    X_df['Country'] = country
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3)
    
    # Treinar modelo
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train.drop(columns=['Country']), y_train)
    
    # Predições
    y_pred = model.predict(X_test.drop(columns=['Country']))
    
    # Análise de fairness
    analyzer = FairnessAnalyzer(
        sensitive_features=['Country'],
        fairness_threshold=0.8
    )
    
    # Analisar todas features
    results = analyzer.analyze_all_features(
        y_test.values,
        y_pred,
        X_test[['Country']]
    )
    
    # Plot
    if 'Country' in results:
        analyzer.plot_fairness_comparison(results['Country'])
    
    # Relatório textual
    report_text = analyzer.generate_fairness_report_text(results)
    print("\n" + report_text)
    
    print("\n✅ Teste completo!")
