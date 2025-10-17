"""Quick product summary utility.

Uso:
    python -m utils.product_summary

Objetivo:
    Fornecer uma visão consolidada dos artefatos finais gerados pelo pipeline,
    facilitando entendimento do "produto" entregue (modelo oficial, thresholds,
    ensemble, governança, monitoramento e explicabilidade).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    'summarize_best_model',
    'summarize_thresholds',
    'summarize_ensemble',
    'file_summary',
    'main',
]

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")

KEY_FILES = [
    "best_model_meta.json",
    "model_card.json",
    "baseline_candidates.json",
    "baseline_models.csv",
    "baseline_metrics_at_k.csv",
    "tuning_results.json",
    "thresholds.json",
    "threshold_analysis.csv",
    "risk_scores_ensemble.csv",
    "ensemble_metadata.json",
    "monitor_feature_shift.csv",
    "monitor_score_shift.csv",
    "monitor_summary.json",
    "validation_report.json",
    "permutation_importance.csv",
    "shap_importance.csv",
    "pipeline_results.json",
    "lineage_registry.json"
]

SEPARATOR = "=" * 78


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not _exists(path):
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_best_model(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Model Name      : {meta.get('model_name')}")
    parts.append(f"Variant         : {meta.get('variant')}")
    parts.append(f"Source Stage    : {meta.get('source')}")
    if meta.get('primary_metric') and meta.get('primary_value') is not None:
        parts.append(f"Primary Metric  : {meta.get('primary_metric')}={meta.get('primary_value'):.4f}")
    if meta.get('decision_action'):
        parts.append(f"Decision Action : {meta.get('decision_action')}")
    if meta.get('improvement_over_baseline') is not None:
        parts.append(f"Δ vs Baseline   : {meta.get('improvement_over_baseline'):.4f}")
    if meta.get('model_file_path'):
        parts.append(f"Model File      : {meta.get('model_file_path')}")
    else:
        # Reconstroi caminho provável
        source = meta.get('source')
        variant = meta.get('variant')
        if source == 'tuning':
            guess = MODELS_DIR / 'best_model_tuned.pkl'
        elif variant == 'core':
            guess = MODELS_DIR / 'best_baseline_core.pkl'
        else:
            guess = MODELS_DIR / 'best_baseline.pkl'
        parts.append(f"(Possível arquivo) : {guess}")
    return "\n".join(parts)


def summarize_thresholds(payload: Dict[str, Any]) -> str:
    best = payload.get('best_threshold')
    metrics = payload.get('best_metrics', {})
    optimize_metric = payload.get('threshold_config', {}).get('optimize_metric')
    flagged = metrics.get('flagged_cases')
    lines = [
        f"Best Threshold : {best}",
        f"Optimize Metric: {optimize_metric}",
    ]
    if optimize_metric in metrics:
        lines.append(f"{optimize_metric}={metrics.get(optimize_metric):.4f}")
    if flagged is not None:
        lines.append(f"Flagged Cases  : {flagged}")
    return "\n".join(lines)


def summarize_ensemble(meta: Dict[str, Any]) -> str:
    comps = ", ".join(meta.get('components_used', []))
    bands = meta.get('band_distribution', {})
    band_str = ", ".join(f"{k}:{v}" for k, v in bands.items()) if bands else "-"
    return (
        f"Components  : {comps}\n"
        f"Blend Method: {meta.get('blend_method')}\n"
        f"Risk Bands  : {band_str}"
    )


def file_summary() -> List[str]:
    rows: List[str] = []
    for fname in KEY_FILES:
        path = ARTIFACTS_DIR / fname
        status = "OK" if _exists(path) else "MISSING"
        size = path.stat().st_size if path.exists() else 0
        rows.append(f"{fname:<30} | {status:<8} | {size:>7} bytes")
    return rows


def main() -> None:
    print(SEPARATOR)
    print("PIPELINE PRODUCT SUMMARY")
    print(SEPARATOR)

    print("\n[1] Arquivos-Chave")
    for line in file_summary():
        print(line)

    # Best model
    meta = _read_json(ARTIFACTS_DIR / 'best_model_meta.json')
    if meta:
        print("\n[2] Modelo Oficial Selecionado")
        print(summarize_best_model(meta))
    else:
        print("\n[2] Modelo Oficial Selecionado\n(best_model_meta.json ausente – execute estágio baselines/tuning)")

    # Thresholds
    thresholds = _read_json(ARTIFACTS_DIR / 'thresholds.json')
    if thresholds:
        print("\n[3] Threshold Otimizado")
        print(summarize_thresholds(thresholds))
    else:
        print("\n[3] Threshold Otimizado\n( thresholds.json ausente – execute estágio thresholds )")

    # Ensemble
    ensemble_meta = _read_json(ARTIFACTS_DIR / 'ensemble_metadata.json')
    if ensemble_meta:
        print("\n[4] Ensemble / Risco")
        print(summarize_ensemble(ensemble_meta))
    else:
        print("\n[4] Ensemble / Risco\n( ensemble_metadata.json ausente – execute estágio ensemble )")

    # Governance / Validation
    validation = _read_json(ARTIFACTS_DIR / 'validation_report.json')
    if validation:
        art_status = validation.get('artifacts_status', {})
        completeness = art_status.get('completeness')
        print("\n[5] Validação & Governança")
        print(f"Artifacts completeness: {completeness}")
        missing = art_status.get('missing') or []
        if missing:
            print(f"Faltando: {', '.join(missing)}")
    else:
        print("\n[5] Validação & Governança\n(validation_report.json ausente – execute estágio validation)")

    # Explainability quick existence
    perm_ok = _exists(ARTIFACTS_DIR / 'permutation_importance.csv')
    shap_ok = _exists(ARTIFACTS_DIR / 'shap_importance.csv')
    print("\n[6] Explainability")
    print(f"Permutation Importance: {'OK' if perm_ok else 'MISSING'}")
    print(f"SHAP Importance       : {'OK' if shap_ok else 'MISSING'}")

    print("\n[7] Próximos Passos Sugeridos")
    suggestions: List[str] = []
    if not meta:
        suggestions.append("Executar: baselines (e tuning se aplicável) para gerar best_model_meta.json")
    if meta and not thresholds:
        suggestions.append("Executar thresholds para definir corte operacional")
    if thresholds and not ensemble_meta:
        suggestions.append("Gerar ensemble para consolidar risco final")
    if ensemble_meta and not validation:
        suggestions.append("Rodar validation para consolidar auditoria")
    if not suggestions:
        suggestions.append("Pipeline completo – avaliar deploy / servir modelo / monitoração contínua")
    for s in suggestions:
        print(f"- {s}")

    print("\n" + SEPARATOR)
    print("Fim do resumo.")


if __name__ == "__main__":  # pragma: no cover
    main()
