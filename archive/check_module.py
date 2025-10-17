import sys
print('Python executable:', sys.executable)

try:
    import src.modeling.aml_modeling
    print('Module loaded successfully')

    # Verificar se a função existe
    trainer_class = src.modeling.aml_modeling.AMLModelTrainer
    has_function = hasattr(trainer_class, 'select_best_aml_model')
    print('Function exists:', has_function)

    if has_function:
        # Verificar a assinatura da função
        import inspect
        sig = inspect.signature(trainer_class.select_best_aml_model)
        print('Function signature:', sig)

        # Verificar se a docstring contém "simplificada"
        doc = trainer_class.select_best_aml_model.__doc__
        if doc and "simplificada" in doc:
            print('✅ Function has been updated with simplified version')
        else:
            print('❌ Function still has old version')

except Exception as e:
    print('Error:', e)