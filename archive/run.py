#!/usr/bin/env python3
"""
🚀 AML Pipeline Launcher
Script principal para executar componentes do sistema AML Pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Executar a API REST"""
    print(f"🚀 Iniciando API em {host}:{port}")
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def run_dashboard():
    """Executar o dashboard Streamlit"""
    print("📊 Iniciando Dashboard Streamlit")
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_dashboard.py"]
    subprocess.run(cmd)

def run_notebook(name: str):
    """Executar notebook específico"""
    notebook_path = f"notebooks/{name}"
    if not Path(notebook_path).exists():
        print(f"❌ Notebook não encontrado: {notebook_path}")
        return

    print(f"📓 Executando notebook: {name}")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--execute", "--inplace", notebook_path]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="AML Pipeline Launcher")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # API
    api_parser = subparsers.add_parser("api", help="Executar API REST")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host para API")
    api_parser.add_argument("--port", type=int, default=8000, help="Porta para API")
    api_parser.add_argument("--reload", action="store_true", help="Recarregar automaticamente")

    # Dashboard
    subparsers.add_parser("dashboard", help="Executar dashboard Streamlit")

    # Notebook
    notebook_parser = subparsers.add_parser("notebook", help="Executar notebook")
    notebook_parser.add_argument("name", help="Nome do notebook (caminho relativo)")

    # Listar componentes
    subparsers.add_parser("list", help="Listar componentes disponíveis")

    args = parser.parse_args()

    if args.command == "api":
        run_api(args.host, args.port, args.reload)
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "notebook":
        run_notebook(args.name)
    elif args.command == "list":
        print("📋 Componentes disponíveis:")
        print("  • api        - API REST (FastAPI)")
        print("  • dashboard  - Dashboard Streamlit")
        print("  • notebook   - Notebooks Jupyter")
        print("\n📖 Para mais informações: python run.py {comando} --help")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()