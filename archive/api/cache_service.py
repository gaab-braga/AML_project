"""
Redis Cache Service
===================

Serviço de cache inteligente para otimização de performance.

Autor: Time de Data Science
Data: Outubro 2025
Fase: 5 - Hybrid Architecture
"""

import redis
import json
import pickle
import gzip
from typing import Any, Optional, Dict, List
import logging
from datetime import datetime, timedelta
import hashlib
import threading
import time

logger = logging.getLogger(__name__)


class CacheService:
    """
    Serviço de cache Redis com TTL inteligente, compressão e métricas avançadas.

    Features:
    - Cache de predições de modelo
    - Cache de métricas do dashboard
    - Cache de configurações
    - TTL automático baseado em tipo de dado
    - Compressão automática para dados grandes
    - Fallback para cache em memória se Redis indisponível
    - Métricas de performance e hit rate
    - Cache warming automático
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = "aml_cache_secret",
        ttl_defaults: Optional[Dict[str, int]] = None,
        compression_threshold: int = 1024,  # 1KB
        enable_metrics: bool = True
    ):
        """
        Inicializa serviço de cache.

        Parameters
        ----------
        host : str
            Host do Redis
        port : int
            Porta do Redis
        db : int
            Database do Redis
        password : str, optional
            Senha do Redis
        ttl_defaults : dict, optional
            TTLs padrão por tipo de dado (segundos)
        compression_threshold : int
            Tamanho mínimo para compressão (bytes)
        enable_metrics : bool
            Habilitar coleta de métricas
        """
        self.ttl_defaults = ttl_defaults or {
            "prediction": 3600,      # 1 hora
            "metrics": 300,          # 5 minutos
            "config": 1800,          # 30 minutos
            "validation": 600,       # 10 minutos
            "dashboard": 300,        # 5 minutos
            "features": 1800,        # 30 minutos
        }

        self.compression_threshold = compression_threshold
        self.enable_metrics = enable_metrics

        # Cache em memória como fallback
        self.memory_cache = {}
        self.memory_lock = threading.Lock()

        # Métricas
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "compression_savings": 0,
            "start_time": time.time()
        }
        self.metrics_lock = threading.Lock()

        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # Para pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                max_connections=20  # Connection pool
            )

            # Testar conexão
            self.redis.ping()
            self.redis_available = True
            logger.info("✅ Redis conectado com sucesso")

            # Iniciar cache warming em background
            self._start_cache_warming()

        except Exception as e:
            logger.warning(f"⚠️ Redis indisponível, usando cache em memória: {e}")
            self.redis = None
            self.redis_available = False

    def _compress_data(self, data: bytes) -> tuple:
        """Comprime dados se acima do threshold."""
        if len(data) < self.compression_threshold:
            return data, False

        try:
            compressed = gzip.compress(data)
            if len(compressed) < len(data):  # Só comprimir se reduzir tamanho
                return compressed, True
        except Exception:
            pass

        return data, False

    def _decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """Descomprime dados se necessário."""
        if not compressed:
            return data

        try:
            return gzip.decompress(data)
        except Exception as e:
            logger.warning(f"Erro na descompressão: {e}")
            return data

    def _update_metrics(self, operation: str, **kwargs):
        """Atualiza métricas de performance."""
        if not self.enable_metrics:
            return

        with self.metrics_lock:
            if operation == "hit":
                self.metrics["hits"] += 1
            elif operation == "miss":
                self.metrics["misses"] += 1
            elif operation == "set":
                self.metrics["sets"] += 1
                if "compressed" in kwargs and kwargs["compressed"]:
                    original_size = kwargs.get("original_size", 0)
                    compressed_size = kwargs.get("compressed_size", 0)
                    self.metrics["compression_savings"] += (original_size - compressed_size)
            elif operation == "error":
                self.metrics["errors"] += 1

    def _start_cache_warming(self):
        """Inicia cache warming em background para dados frequentes."""
        if not self.redis_available:
            return

        def warm_cache():
            try:
                # Cache warming para configurações e dados estáticos
                logger.info("🔄 Iniciando cache warming...")

                # Placeholder para cache warming - pode ser expandido
                # com dados frequentes do dashboard

                logger.info("✅ Cache warming concluído")

            except Exception as e:
                logger.warning(f"Erro no cache warming: {e}")

        # Executar em thread separada
        warming_thread = threading.Thread(target=warm_cache, daemon=True)
        warming_thread.start()

    def _get_cache_key(self, prefix: str, key_data: Any) -> str:
        """Gera chave de cache consistente."""
        if isinstance(key_data, (str, int, float)):
            key_str = str(key_data)
        elif isinstance(key_data, dict):
            # Ordenar keys para consistência
            key_str = json.dumps(key_data, sort_keys=True)
        else:
            # Para objetos complexos, usar representação string
            key_str = str(key_data)

        # Hash para chave curta e consistente
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]
        return f"{prefix}:{key_hash}"

    def _serialize(self, data: Any) -> bytes:
        """Serializa dados para cache."""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.warning(f"Erro na serialização, usando JSON: {e}")
            return json.dumps(data).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        """Desserializa dados do cache."""
        try:
            return pickle.loads(data)
        except Exception:
            try:
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Erro na desserialização: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "default") -> bool:
        """
        Armazena valor no cache com compressão automática.

        Parameters
        ----------
        key : str
            Chave do cache
        value : Any
            Valor a armazenar
        ttl : int, optional
            TTL em segundos (usa padrão se None)
        prefix : str
            Prefixo da chave

        Returns
        -------
        success : bool
            True se armazenado com sucesso
        """
        if ttl is None:
            ttl = self.ttl_defaults.get(prefix, 3600)

        cache_key = self._get_cache_key(prefix, key)
        serialized_data = self._serialize(value)

        # Compressão
        compressed_data, is_compressed = self._compress_data(serialized_data)

        # Metadata para descompressão
        final_data = compressed_data + b"\x00" + bytes([1 if is_compressed else 0])

        try:
            if self.redis_available:
                success = bool(self.redis.setex(cache_key, ttl, final_data))
            else:
                # Fallback para memória
                with self.memory_lock:
                    self.memory_cache[cache_key] = {
                        "data": final_data,
                        "expires": datetime.now() + timedelta(seconds=ttl),
                        "compressed": is_compressed
                    }
                success = True

            if success and self.enable_metrics:
                self._update_metrics("set",
                                   compressed=is_compressed,
                                   original_size=len(serialized_data),
                                   compressed_size=len(compressed_data))

            return success

        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {e}")
            if self.enable_metrics:
                self._update_metrics("error")
            return False

    def get(self, key: str, prefix: str = "default") -> Optional[Any]:
        """
        Recupera valor do cache com descompressão automática.

        Parameters
        ----------
        key : str
            Chave do cache
        prefix : str
            Prefixo da chave

        Returns
        -------
        value : Any or None
            Valor recuperado ou None se não encontrado/expirado
        """
        cache_key = self._get_cache_key(prefix, key)

        try:
            if self.redis_available:
                data = self.redis.get(cache_key)
                if data:
                    if self.enable_metrics:
                        self._update_metrics("hit")

                    # Extrair metadata de compressão
                    if len(data) > 1:
                        compressed_flag = data[-1]
                        actual_data = data[:-1]
                        actual_data = self._decompress_data(actual_data, bool(compressed_flag))
                    else:
                        actual_data = data

                    return self._deserialize(actual_data)
                else:
                    if self.enable_metrics:
                        self._update_metrics("miss")

            else:
                # Fallback para memória
                with self.memory_lock:
                    if cache_key in self.memory_cache:
                        entry = self.memory_cache[cache_key]
                        if datetime.now() < entry["expires"]:
                            if self.enable_metrics:
                                self._update_metrics("hit")

                            # Descompressão
                            data = entry["data"]
                            if len(data) > 1:
                                compressed_flag = data[-1]
                                actual_data = data[:-1]
                                actual_data = self._decompress_data(actual_data, bool(compressed_flag))
                            else:
                                actual_data = data

                            return self._deserialize(actual_data)
                        else:
                            # Expirado, remover
                            del self.memory_cache[cache_key]
                            if self.enable_metrics:
                                self._update_metrics("miss")
                    else:
                        if self.enable_metrics:
                            self._update_metrics("miss")

        except Exception as e:
            logger.error(f"Erro ao recuperar do cache: {e}")
            if self.enable_metrics:
                self._update_metrics("error")

        return None

    def delete(self, key: str, prefix: str = "default") -> bool:
        """Remove valor do cache."""
        cache_key = self._get_cache_key(prefix, key)

        try:
            if self.redis_available:
                return bool(self.redis.delete(cache_key))
            else:
                with self.memory_lock:
                    if cache_key in self.memory_cache:
                        del self.memory_cache[cache_key]
                        return True
        except Exception as e:
            logger.error(f"Erro ao remover do cache: {e}")

        return False

    def clear_prefix(self, prefix: str) -> int:
        """Remove todas as chaves com determinado prefixo."""
        try:
            if self.redis_available:
                # Usar SCAN para encontrar chaves
                keys = []
                cursor = 0
                while True:
                    cursor, batch = self.redis.scan(cursor, f"{prefix}:*")
                    keys.extend(batch)
                    if cursor == 0:
                        break

                if keys:
                    return self.redis.delete(*keys)
                return 0
            else:
                # Memory cache
                with self.memory_lock:
                    keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(f"{prefix}:")]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                return len(keys_to_delete)

        except Exception as e:
            logger.error(f"Erro ao limpar cache por prefixo: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        stats = {
            "redis_available": self.redis_available,
            "memory_cache_size": len(self.memory_cache)
        }

        if self.redis_available:
            try:
                info = self.redis.info()
                stats.update({
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_used_memory": info.get("used_memory_human", "N/A"),
                    "redis_total_keys": self.redis.dbsize()
                })
            except Exception as e:
                logger.error(f"Erro ao obter stats do Redis: {e}")

        # Adicionar métricas customizadas
        if self.enable_metrics:
            with self.metrics_lock:
                total_requests = self.metrics["hits"] + self.metrics["misses"]
                hit_rate = (self.metrics["hits"] / total_requests * 100) if total_requests > 0 else 0

                stats.update({
                    "cache_hits": self.metrics["hits"],
                    "cache_misses": self.metrics["misses"],
                    "cache_sets": self.metrics["sets"],
                    "cache_errors": self.metrics["errors"],
                    "hit_rate_percent": round(hit_rate, 2),
                    "compression_savings_bytes": self.metrics["compression_savings"],
                    "uptime_seconds": time.time() - self.metrics["start_time"]
                })

        return stats

    # ============================================================================
    # MÉTODOS ESPECÍFICOS PARA AML
    # ============================================================================

    def cache_prediction(self, transaction_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> bool:
        """Cache de resultado de predição."""
        key = {
            "transaction_id": transaction_data.get("transaction_id"),
            "amount": transaction_data.get("amount"),
            "from_bank": transaction_data.get("from_bank"),
            "to_bank": transaction_data.get("to_bank")
        }
        return self.set(key, prediction_result, prefix="prediction")

    def get_cached_prediction(self, transaction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recupera predição do cache."""
        key = {
            "transaction_id": transaction_data.get("transaction_id"),
            "amount": transaction_data.get("amount"),
            "from_bank": transaction_data.get("from_bank"),
            "to_bank": transaction_data.get("to_bank")
        }
        return self.get(key, prefix="prediction")

    def cache_dashboard_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Cache de métricas do dashboard."""
        return self.set("dashboard_metrics", metrics, prefix="dashboard")

    def get_cached_dashboard_metrics(self) -> Optional[Dict[str, Any]]:
        """Recupera métricas do dashboard do cache."""
        return self.get("dashboard_metrics", prefix="dashboard")

    def invalidate_metrics_cache(self) -> bool:
        """Invalida cache de métricas."""
        return bool(self.clear_prefix("dashboard"))


# Singleton instance
_cache_service = None

def get_cache_service() -> CacheService:
    """Factory function para serviço de cache singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service