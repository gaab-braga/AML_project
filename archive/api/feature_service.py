"""
Feature Engineering Service
============================

Calcula features em tempo real para scoring via API.

Autor: Time de Data Science
Data: Outubro 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Serviço para feature engineering em tempo real.
    
    Rápido e otimizado para latência < 50ms.
    """
    
    def __init__(
        self,
        label_encoders_path: Optional[str] = None,
        freq_mappings_path: Optional[str] = None
    ):
        """
        Inicializa serviço.
        
        Parameters
        ----------
        label_encoders_path : str, optional
            Caminho dos label encoders
        freq_mappings_path : str, optional
            Caminho dos frequency mappings
        """
        self.label_encoders = None
        self.freq_mappings = None
        
        # Carregar label encoders se disponível
        if label_encoders_path is None:
            label_encoders_path = Path("data/label_encoders.pkl")
        
        if Path(label_encoders_path).exists():
            try:
                with open(label_encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                logger.info(f"✅ Label encoders carregados: {len(self.label_encoders)} features")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar label encoders: {e}")
        
        # Carregar frequency mappings se disponível
        if freq_mappings_path is None:
            freq_mappings_path = Path("data/freq_mappings.pkl")
        
        if Path(freq_mappings_path).exists():
            try:
                with open(freq_mappings_path, 'rb') as f:
                    self.freq_mappings = pickle.load(f)
                logger.info(f"✅ Frequency mappings carregados: {len(self.freq_mappings)} features")
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar freq mappings: {e}")
    
    def engineer_features(self, transaction: Dict[str, Any]) -> pd.DataFrame:
        """
        Calcula features para uma transação.
        
        Parameters
        ----------
        transaction : Dict
            Dicionário com campos da transação
            
        Returns
        -------
        features : pd.DataFrame
            DataFrame com features (1 linha)
        """
        features = {}
        
        # ==========================================
        # 1. Features Originais
        # ==========================================
        
        features['Amount Paid'] = transaction.get('amount', 0.0)
        features['From Bank'] = transaction.get('from_bank', '')
        features['To Bank'] = transaction.get('to_bank', '')
        features['Account'] = transaction.get('account', '')
        
        # ==========================================
        # 2. Features Temporais
        # ==========================================
        
        timestamp = transaction.get('timestamp')
        
        if timestamp:
            if isinstance(timestamp, str):
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    # Fallback: usar datetime atual
                    dt = datetime.now()
            else:
                dt = timestamp
            
            # Extrair componentes temporais
            features['Receiving Currency_hour'] = dt.hour
            features['Receiving Currency_day'] = dt.day
            features['Receiving Currency_month'] = dt.month
            features['Receiving Currency_dayofweek'] = dt.weekday()
            features['Receiving Currency_is_weekend'] = 1 if dt.weekday() >= 5 else 0
            features['Receiving Currency_quarter'] = (dt.month - 1) // 3 + 1
            
            # Períodos do dia
            hour = dt.hour
            features['Receiving Currency_is_night'] = 1 if (hour >= 22 or hour < 6) else 0
            features['Receiving Currency_is_business_hours'] = 1 if (9 <= hour < 18) else 0
        else:
            # Usar hora atual se não fornecido
            dt = datetime.now()
            features['Receiving Currency_hour'] = dt.hour
            features['Receiving Currency_day'] = dt.day
            features['Receiving Currency_month'] = dt.month
            features['Receiving Currency_dayofweek'] = dt.weekday()
            features['Receiving Currency_is_weekend'] = 1 if dt.weekday() >= 5 else 0
            features['Receiving Currency_quarter'] = (dt.month - 1) // 3 + 1
            features['Receiving Currency_is_night'] = 1 if (dt.hour >= 22 or dt.hour < 6) else 0
            features['Receiving Currency_is_business_hours'] = 1 if (9 <= dt.hour < 18) else 0
        
        # ==========================================
        # 3. Features de Valor
        # ==========================================
        
        amount = features['Amount Paid']
        
        features['Amount Paid_log'] = np.log1p(amount)
        features['Amount Paid_squared'] = amount ** 2
        features['Amount Paid_is_high'] = 1 if amount > 10000 else 0
        features['Amount Paid_is_round'] = 1 if (amount % 1000 == 0) else 0
        
        # Bins de valor
        if amount < 1000:
            features['Amount Paid_bin'] = 0
        elif amount < 5000:
            features['Amount Paid_bin'] = 1
        elif amount < 10000:
            features['Amount Paid_bin'] = 2
        else:
            features['Amount Paid_bin'] = 3
        
        # ==========================================
        # 4. Features Categóricas Encoded
        # ==========================================
        
        if self.label_encoders:
            # From Bank
            if 'From Bank' in self.label_encoders:
                from_bank = transaction.get('from_bank', '')
                le = self.label_encoders['From Bank']
                if from_bank in le.classes_:
                    features['From Bank_encoded'] = le.transform([from_bank])[0]
                else:
                    # Categoria desconhecida (usar -1 ou classe padrão)
                    features['From Bank_encoded'] = -1
            
            # To Bank
            if 'To Bank' in self.label_encoders:
                to_bank = transaction.get('to_bank', '')
                le = self.label_encoders['To Bank']
                if to_bank in le.classes_:
                    features['To Bank_encoded'] = le.transform([to_bank])[0]
                else:
                    features['To Bank_encoded'] = -1
            
            # Account
            if 'Account' in self.label_encoders:
                account = transaction.get('account', '')
                le = self.label_encoders['Account']
                if account in le.classes_:
                    features['Account_encoded'] = le.transform([account])[0]
                else:
                    features['Account_encoded'] = -1
        
        # ==========================================
        # 5. Frequency Encoding
        # ==========================================
        
        if self.freq_mappings:
            if 'From Bank' in self.freq_mappings:
                from_bank = transaction.get('from_bank', '')
                features['From Bank_freq'] = self.freq_mappings['From Bank'].get(from_bank, 0.0)
            
            if 'To Bank' in self.freq_mappings:
                to_bank = transaction.get('to_bank', '')
                features['To Bank_freq'] = self.freq_mappings['To Bank'].get(to_bank, 0.0)
            
            if 'Account' in self.freq_mappings:
                account = transaction.get('account', '')
                features['Account_freq'] = self.freq_mappings['Account'].get(account, 0.0)
        
        # ==========================================
        # 6. Features de Interação
        # ==========================================
        
        # Same bank transfer
        from_bank = transaction.get('from_bank', '')
        to_bank = transaction.get('to_bank', '')
        features['is_same_bank'] = 1 if from_bank == to_bank else 0
        
        # Amount x hour interaction
        features['amount_x_hour'] = amount * features['Receiving Currency_hour']
        
        # ==========================================
        # 7. Campos Opcionais
        # ==========================================
        
        # Payment Currency
        payment_currency = transaction.get('payment_currency')
        if payment_currency:
            features['Payment Currency'] = payment_currency
            
            if self.label_encoders and 'Payment Currency' in self.label_encoders:
                le = self.label_encoders['Payment Currency']
                if payment_currency in le.classes_:
                    features['Payment Currency_encoded'] = le.transform([payment_currency])[0]
                else:
                    features['Payment Currency_encoded'] = -1
        
        # Receiving Currency
        receiving_currency = transaction.get('receiving_currency')
        if receiving_currency:
            features['Receiving Currency'] = receiving_currency
            
            if self.label_encoders and 'Receiving Currency' in self.label_encoders:
                le = self.label_encoders['Receiving Currency']
                if receiving_currency in le.classes_:
                    features['Receiving Currency_encoded'] = le.transform([receiving_currency])[0]
                else:
                    features['Receiving Currency_encoded'] = -1
        
        # Payment Format
        payment_format = transaction.get('payment_format')
        if payment_format:
            features['Payment Format'] = payment_format
            
            if self.label_encoders and 'Payment Format' in self.label_encoders:
                le = self.label_encoders['Payment Format']
                if payment_format in le.classes_:
                    features['Payment Format_encoded'] = le.transform([payment_format])[0]
                else:
                    features['Payment Format_encoded'] = -1
        
        # ==========================================
        # Converter para DataFrame
        # ==========================================
        
        df = pd.DataFrame([features])
        
        return df
    
    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Valida schema de features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
            
        Returns
        -------
        is_valid : bool
            True se válido
        """
        # Verificações básicas
        if X.empty:
            logger.error("❌ DataFrame vazio")
            return False
        
        if X.isnull().any().any():
            logger.warning("⚠️ Features com valores nulos")
            # Preencher com 0 ou -1
            X.fillna(0, inplace=True)
        
        # Features obrigatórias (mínimo)
        required_features = [
            'Amount Paid',
            'Receiving Currency_hour',
            'Amount Paid_log'
        ]
        
        missing = [f for f in required_features if f not in X.columns]
        
        if missing:
            logger.error(f"❌ Features obrigatórias faltando: {missing}")
            return False
        
        return True
    
    def batch_engineer_features(
        self,
        transactions: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calcula features para múltiplas transações.
        
        Parameters
        ----------
        transactions : List[Dict]
            Lista de transações
            
        Returns
        -------
        features : pd.DataFrame
            DataFrame com features (N linhas)
        """
        all_features = []
        
        for transaction in transactions:
            features = self.engineer_features(transaction)
            all_features.append(features)
        
        # Concatenar
        df = pd.concat(all_features, ignore_index=True)
        
        return df


# Teste
if __name__ == "__main__":
    print("="*80)
    print("TESTE: Feature Engineering Service")
    print("="*80)
    
    service = FeatureService()
    
    # Transação de teste
    transaction = {
        'amount': 5000.0,
        'from_bank': 'HSBC',
        'to_bank': 'Deutsche Bank',
        'account': 'ACC123456',
        'timestamp': '2024-10-25T14:30:00'
    }
    
    print("\n📦 Transação de teste:")
    print(transaction)
    
    # Calcular features
    features = service.engineer_features(transaction)
    
    print(f"\n✅ Features calculadas: {features.shape[1]} colunas")
    print("\nAmostra de features:")
    print(features.head().T[:10])
    
    # Validar
    is_valid = service.validate_features(features)
    print(f"\nValidação: {'✅ OK' if is_valid else '❌ ERRO'}")
    
    # Batch
    transactions = [transaction] * 10
    batch_features = service.batch_engineer_features(transactions)
    print(f"\nBatch features: {batch_features.shape}")
    
    print("\n✅ Teste concluído!")
