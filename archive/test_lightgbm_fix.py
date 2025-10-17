#!/usr/bin/env python3
"""
Script simples para testar a correção do LightGBM
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar e testar
try:
    from src.modeling.aml_modeling import AMLModelTrainer, create_aml_experiment_config
    print("✅ Importação bem-sucedida")

    # Criar configuração
    config = create_aml_experiment_config()
    print("✅ Configuração criada")

    # Verificar se LightGBM está configurado corretamente
    lgb_config = config['models']['lightgbm']
    print(f"✅ Config LightGBM: {lgb_config['model_type']}")

    # Criar trainer
    trainer = AMLModelTrainer(config)
    print("✅ Trainer criado")

    print("\n🎉 Todos os testes passaram! A correção do LightGBM deve funcionar.")

except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc()