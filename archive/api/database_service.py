"""
Database Optimization Service
=============================

Serviço para otimização de queries e connection pooling.

Autor: Time de Data Science
Data: Outubro 2025
Fase: 4.3 - Backend Optimization
"""

import sqlite3
import threading
import time
from typing import Optional, Any, Dict, List
import logging
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """
    Connection pool para SQLite com thread safety.
    
    Mantém múltiplas conexões para melhor performance em ambientes concorrentes.
    """
    
    def __init__(self, db_path: str, max_connections: int = 10):
        """
        Inicializa pool de conexões.
        
        Parameters
        ----------
        db_path : str
            Caminho do banco de dados
        max_connections : int
            Número máximo de conexões no pool
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        self._initialized = False
        
        # Inicializar pool
        self._init_pool()
    
    def _init_pool(self):
        """Inicializa pool de conexões."""
        with self.lock:
            for _ in range(self.max_connections):
                conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0
                )
                # Otimizações SQLite
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA cache_size=10000;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
                
                self.connections.append({
                    "connection": conn,
                    "in_use": False,
                    "last_used": time.time()
                })
            
            self._initialized = True
            logger.info(f"✅ Pool de conexões inicializado: {self.max_connections} conexões")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Obtém conexão do pool.
        
        Returns
        -------
        connection : sqlite3.Connection
            Conexão disponível
        """
        with self.lock:
            # Procurar conexão disponível
            for conn_info in self.connections:
                if not conn_info["in_use"]:
                    conn_info["in_use"] = True
                    conn_info["last_used"] = time.time()
                    return conn_info["connection"]
            
            # Se não encontrou, usar a menos recentemente usada
            oldest_conn = min(self.connections, key=lambda x: x["last_used"])
            oldest_conn["in_use"] = True
            oldest_conn["last_used"] = time.time()
            return oldest_conn["connection"]
    
    def release_connection(self, connection: sqlite3.Connection):
        """Libera conexão de volta ao pool."""
        with self.lock:
            for conn_info in self.connections:
                if conn_info["connection"] is connection:
                    conn_info["in_use"] = False
                    break
    
    def close_all(self):
        """Fecha todas as conexões."""
        with self.lock:
            for conn_info in self.connections:
                try:
                    conn_info["connection"].close()
                except Exception as e:
                    logger.error(f"Erro ao fechar conexão: {e}")
            
            self.connections.clear()
            logger.info("🛑 Todas as conexões fechadas")


class OptimizedDatabaseService:
    """
    Serviço de banco de dados otimizado com connection pooling e queries eficientes.
    """
    
    def __init__(self, db_path: str):
        """
        Inicializa serviço de banco otimizado.
        
        Parameters
        ----------
        db_path : str
            Caminho do banco de dados
        """
        self.db_path = Path(db_path)
        self.pool = DatabaseConnectionPool(str(self.db_path))
        
        # Criar tabelas se não existirem
        self._init_tables()
    
    def _init_tables(self):
        """Inicializa tabelas necessárias para a API."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tabela de transações processadas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    amount REAL NOT NULL,
                    from_bank TEXT,
                    to_bank TEXT,
                    account TEXT,
                    account_1 TEXT,
                    risk_score REAL,
                    decision TEXT,
                    model_version TEXT,
                    latency_ms REAL,
                    features_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Índices para performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_risk_score ON transactions(risk_score);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_decision ON transactions(decision);")
            
            # Tabela de resultados de validação
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    issues_found INTEGER DEFAULT 0,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_results(timestamp);")
            
            # Tabela de configurações
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT,
                    description TEXT,
                    last_updated TEXT NOT NULL,
                    updated_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Inserir configurações padrão
            default_configs = [
                ("alert_threshold", "0.8", "Threshold for triggering AML alerts", "2025-10-01T00:00:00", "system"),
                ("model_version", "v2.1.0", "Current model version in production", "2025-10-01T00:00:00", "system"),
                ("cache_ttl_predictions", "3600", "Cache TTL for predictions (seconds)", "2025-10-01T00:00:00", "system"),
                ("cache_ttl_metrics", "300", "Cache TTL for dashboard metrics (seconds)", "2025-10-01T00:00:00", "system"),
            ]
            
            for key, value, desc, updated, user in default_configs:
                cursor.execute("""
                    INSERT OR IGNORE INTO config (key, value, description, last_updated, updated_by)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, value, desc, updated, user))
            
            conn.commit()
            logger.info("✅ Tabelas de API inicializadas")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager para obter conexão do pool.
        
        Usage:
            with db_service.get_connection() as conn:
                cursor = conn.cursor()
                # use connection
        """
        conn = self.pool.get_connection()
        try:
            yield conn
        finally:
            self.pool.release_connection(conn)
    
    def execute_query(self, query: str, params: tuple = (), fetch: bool = True) -> Optional[List[tuple]]:
        """
        Executa query otimizada.
        
        Parameters
        ----------
        query : str
            Query SQL
        params : tuple
            Parâmetros da query
        fetch : bool
            Se deve buscar resultados
            
        Returns
        -------
        results : List[tuple] or None
            Resultados da query ou None se não fetch
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute(query, params)
                
                if fetch:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return None
                    
            except Exception as e:
                logger.error(f"Erro na query: {query} - {e}")
                conn.rollback()
                raise
    
    def insert_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Insere transação processada."""
        query = """
            INSERT OR REPLACE INTO transactions 
            (transaction_id, timestamp, amount, from_bank, to_bank, account, account_1, 
             risk_score, decision, model_version, latency_ms, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            transaction_data["transaction_id"],
            transaction_data["timestamp"],
            transaction_data["amount"],
            transaction_data.get("from_bank"),
            transaction_data.get("to_bank"),
            transaction_data.get("account"),
            transaction_data.get("account_1"),
            transaction_data.get("risk_score"),
            transaction_data.get("decision"),
            transaction_data.get("model_version"),
            transaction_data.get("latency_ms"),
            str(transaction_data.get("features", {}))
        )
        
        try:
            self.execute_query(query, params, fetch=False)
            return True
        except Exception as e:
            logger.error(f"Erro ao inserir transação: {e}")
            return False
    
    def get_dashboard_metrics_optimized(self) -> Dict[str, Any]:
        """
        Busca métricas do dashboard com queries otimizadas.
        
        Returns
        -------
        metrics : dict
            Métricas calculadas
        """
        # Query otimizada com CTE para calcular tudo em uma passada
        query = """
            WITH daily_stats AS (
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(CASE WHEN risk_score >= 0.8 THEN 1 ELSE 0 END) as alerts_today,
                    SUM(CASE WHEN risk_score < 0.3 THEN 1 ELSE 0 END) as low_risk,
                    SUM(CASE WHEN risk_score >= 0.3 AND risk_score < 0.7 THEN 1 ELSE 0 END) as medium_risk,
                    SUM(CASE WHEN risk_score >= 0.7 THEN 1 ELSE 0 END) as high_risk
                FROM transactions 
                WHERE DATE(timestamp) = DATE('now')
            )
            SELECT * FROM daily_stats;
        """
        
        results = self.execute_query(query)
        if results:
            row = results[0]
            return {
                "total_transactions": row[0],
                "alerts_today": row[1],
                "risk_distribution": {
                    "low": row[2],
                    "medium": row[3], 
                    "high": row[4]
                }
            }
        
        return {
            "total_transactions": 0,
            "alerts_today": 0,
            "risk_distribution": {"low": 0, "medium": 0, "high": 0}
        }
    
    def get_transactions_paginated(
        self, 
        limit: int = 100, 
        offset: int = 0,
        risk_threshold: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Busca transações com paginação otimizada.
        
        Returns
        -------
        result : dict
            Transações e metadados de paginação
        """
        # Query base
        query = """
            SELECT transaction_id, timestamp, amount, from_bank, to_bank, 
                   risk_score, decision, model_version, latency_ms
            FROM transactions
            WHERE 1=1
        """
        count_query = "SELECT COUNT(*) FROM transactions WHERE 1=1"
        params = []
        
        # Filtros
        if risk_threshold is not None:
            query += " AND risk_score >= ?"
            count_query += " AND risk_score >= ?"
            params.append(risk_threshold)
        
        if start_date:
            query += " AND DATE(timestamp) >= ?"
            count_query += " AND DATE(timestamp) >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(timestamp) <= ?"
            count_query += " AND DATE(timestamp) <= ?"
            params.append(end_date)
        
        # Ordenação e paginação
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        # Executar queries
        transactions = self.execute_query(query, tuple(params))
        total_count = self.execute_query(count_query, tuple(params[:-2]))[0][0]
        
        # Formatar resultados
        transaction_list = []
        for row in transactions:
            transaction_list.append({
                "transaction_id": row[0],
                "timestamp": row[1],
                "amount": row[2],
                "from_bank": row[3],
                "to_bank": row[4],
                "risk_score": row[5],
                "decision": row[6],
                "model_version": row[7],
                "latency_ms": row[8]
            })
        
        return {
            "transactions": transaction_list,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Remove dados antigos para manter performance."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        queries = [
            "DELETE FROM transactions WHERE timestamp < ?",
            "DELETE FROM validation_results WHERE timestamp < ?",
        ]
        
        for query in queries:
            try:
                self.execute_query(query, (cutoff_date,), fetch=False)
                logger.info(f"✅ Dados antigos removidos (cutoff: {cutoff_date})")
            except Exception as e:
                logger.error(f"Erro ao limpar dados antigos: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do banco."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Estatísticas das tabelas
            stats = {}
            
            tables = ["transactions", "validation_results", "config"]
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT COUNT(name) FROM pragma_index_list('{table}')")
                    indexes = cursor.fetchone()[0]
                    
                    stats[table] = {
                        "row_count": count,
                        "indexes": indexes
                    }
                except Exception as e:
                    logger.error(f"Erro ao obter stats da tabela {table}: {e}")
                    stats[table] = {"error": str(e)}
            
            return stats
        
    def initialize_database(self):
        """Inicializa banco de dados (compatibilidade com testes)."""
        self._init_tables()
        logger.info("✅ Banco de dados inicializado")


# Singleton instance
_db_service = None

def get_database_service(db_path: str = "data/dashboard.db") -> OptimizedDatabaseService:
    """Factory function para serviço de banco otimizado singleton."""
    global _db_service
    if _db_service is None:
        _db_service = OptimizedDatabaseService(db_path)
    return _db_service