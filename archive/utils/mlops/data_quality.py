"""
Data Quality Monitoring
=======================

Valida qualidade de dados de entrada em produção.

Autor: Time de Data Science  
Data: Outubro 2025
Fase: 4.2 - Qualidade de Dados
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Relatório de qualidade de dados."""
    passed: bool
    issues: List[Dict]
    warnings: List[Dict]
    summary: Dict


class DataQualityValidator:
    """Validador de qualidade de dados."""
    
    def __init__(self, missing_threshold: float = 0.1, outlier_std: float = 5.0):
        self.missing_threshold = missing_threshold
        self.outlier_std = outlier_std
        self.schema_: Optional[Dict] = None
        self.stats_: Optional[Dict] = None
        
    def fit(self, X: pd.DataFrame):
        """Aprende schema e estatísticas de referência."""
        self.schema_ = {col: {'dtype': str(X[col].dtype), 'nullable': X[col].isna().any()} for col in X.columns}
        self.stats_ = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.stats_[col] = {'mean': X[col].mean(), 'std': X[col].std(), 'min': X[col].min(), 'max': X[col].max()}
        logger.info(f"✅ Schema aprendido: {len(X.columns)} colunas")
        
    def validate(self, X: pd.DataFrame) -> DataQualityReport:
        """Valida dados."""
        if self.schema_ is None:
            raise ValueError("Validator não foi treinado.")
        
        issues = []
        warnings = []
        
        # Schema validation
        missing_cols = set(self.schema_.keys()) - set(X.columns)
        if missing_cols:
            issues.append({'type': 'missing_columns', 'columns': list(missing_cols)})
        
        # Missing values
        for col in X.columns:
            missing_pct = X[col].isna().mean()
            if missing_pct > self.missing_threshold:
                issues.append({'type': 'high_missing', 'column': col, 'missing_pct': missing_pct})
        
        passed = len(issues) == 0
        return DataQualityReport(passed=passed, issues=issues, warnings=warnings, summary={'total_issues': len(issues)})


if __name__ == "__main__":
    print("Teste: Data Quality")
    X_train = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['x', 'y', 'z', 'x', 'y']})
    validator = DataQualityValidator()
    validator.fit(X_train)
    X_prod = pd.DataFrame({'A': [1, 2, None, 100, 5], 'B': ['x', 'y', 'z', 'x', 'y']})
    report = validator.validate(X_prod)
    print(f"Passed: {report.passed}, Issues: {len(report.issues)}")
    print("✅ Teste concluído!")
