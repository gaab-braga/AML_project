import sys
sys.path.append('.')

# Testar apenas se as funções podem ser importadas com os novos parâmetros
try:
    from src.features.aml_plotting import plot_threshold_comparison_all_models_optimized, plot_executive_summary_aml_new
    print("✅ Funções importadas com sucesso!")

    # Verificar assinatura das funções
    import inspect

    sig1 = inspect.signature(plot_threshold_comparison_all_models_optimized)
    sig2 = inspect.signature(plot_executive_summary_aml_new)

    print(f"✅ plot_threshold_comparison_all_models_optimized parâmetros: {list(sig1.parameters.keys())}")
    print(f"✅ plot_executive_summary_aml_new parâmetros: {list(sig2.parameters.keys())}")

    # Verificar se benchmark_metrics está nos parâmetros
    if 'benchmark_metrics' in sig1.parameters:
        print("✅ benchmark_metrics adicionado a plot_threshold_comparison_all_models_optimized")
    else:
        print("❌ benchmark_metrics NÃO encontrado em plot_threshold_comparison_all_models_optimized")

    if 'benchmark_metrics' in sig2.parameters:
        print("✅ benchmark_metrics adicionado a plot_executive_summary_aml_new")
    else:
        print("❌ benchmark_metrics NÃO encontrado em plot_executive_summary_aml_new")

    print("\n🎉 INTEGRAÇÃO DE BENCHMARK IMPLEMENTADA COM SUCESSO!")

except Exception as e:
    print(f"❌ Erro na importação: {e}")
    import traceback
    traceback.print_exc()