"""
Factory para criação de pipelines ML para detecção AML.
Suporta múltiplos algoritmos com configuração unificada.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb


class PipelineFactory:
    """
    Factory para criar pipelines ML configurados.
    
    Suporta XGBoost e LightGBM com pré-processamento padronizado:
    - Imputação de valores ausentes (mediana)
    - Normalização (MinMaxScaler)
    - Classificador configurável
    """
    
    SUPPORTED_MODELS = {
        'xgb': xgb.XGBClassifier,
        'lgbm': lgb.LGBMClassifier
    }
    
    @classmethod
    def create(cls, model_name: str, model_params: dict) -> Pipeline:
        """
        Cria pipeline completo para o modelo especificado.
        
        Args:
            model_name: Nome do modelo ('xgb' ou 'lgbm')
            model_params: Dicionário de hiperparâmetros do modelo
            
        Returns:
            Pipeline sklearn configurado
            
        Raises:
            ValueError: Se modelo não for suportado
            
        Examples:
            >>> params = {'n_estimators': 100, 'max_depth': 5}
            >>> pipeline = PipelineFactory.create('xgb', params)
            >>> pipeline.fit(X_train, y_train)
        """
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Modelo '{model_name}' não suportado. "
                f"Modelos disponíveis: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        classifier_class = cls.SUPPORTED_MODELS[model_name]
        classifier = classifier_class(**model_params)
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler()),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    @classmethod
    def get_supported_models(cls):
        """Retorna lista de modelos suportados."""
        return list(cls.SUPPORTED_MODELS.keys())


# Função de compatibilidade com código legado
def create_pipeline(model_name: str, model_params: dict) -> Pipeline:
    """
    Wrapper de compatibilidade para código existente.
    Use PipelineFactory.create() para novo código.
    """
    return PipelineFactory.create(model_name, model_params)
