"""
Reporting Module - Executive-level reporting and business intelligence
"""

from .executive_summary import (
    calculate_baseline_costs,
    calculate_ml_impact,
    calculate_roi_metrics,
    generate_financial_dashboard,
    generate_roi_analysis,
    generate_risk_assessment,
    generate_implementation_roadmap,
    generate_executive_dashboard,
    print_executive_summary
)

__all__ = [
    'calculate_baseline_costs',
    'calculate_ml_impact',
    'calculate_roi_metrics',
    'generate_financial_dashboard',
    'generate_roi_analysis',
    'generate_risk_assessment',
    'generate_implementation_roadmap',
    'generate_executive_dashboard',
    'print_executive_summary'
]
