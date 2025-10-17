"""
Security Service - Autenticação e Autorização
==============================================

Sistema completo de segurança para AML Dashboard.

Features:
- RBAC (Role-Based Access Control) avançado
- Autenticação JWT com refresh tokens
- Encriptação de dados sensíveis
- Logs de auditoria detalhados
- Rate limiting e proteção contra ataques

Autor: Time de Data Science
Data: Outubro 2025
Fase: 4.4 - Segurança e Compliance
"""

import hashlib
import secrets
import bcrypt
import jwt
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import sqlite3
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os

logger = logging.getLogger(__name__)


class SecurityService:
    """
    Serviço centralizado de segurança e compliance.

    Gerencia autenticação, autorização, encriptação e auditoria.
    """

    def __init__(
        self,
        jwt_secret: str = None,
        encryption_key: str = None,
        db_path: str = "data/dashboard.db"
    ):
        """
        Inicializa serviço de segurança.

        Parameters
        ----------
        jwt_secret : str, optional
            Chave secreta para JWT (gerada automaticamente se None)
        encryption_key : str, optional
            Chave para encriptação (gerada automaticamente se None)
        db_path : str
            Caminho do banco de dados
        """
        self.jwt_secret = jwt_secret or self._generate_jwt_secret()
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.db_path = Path(db_path)
        self.fernet = Fernet(self.encryption_key)

        # Configurações de segurança
        self.jwt_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=7)
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)

        # Roles e permissões padrão
        self.default_roles = {
            "admin": {
                "name": "Administrator",
                "permissions": [
                    "user:read", "user:write", "user:delete",
                    "config:read", "config:write",
                    "audit:read",
                    "dashboard:read", "dashboard:write",
                    "api:read", "api:write",
                    "security:read", "security:write"
                ],
                "level": 100
            },
            "analyst": {
                "name": "AML Analyst",
                "permissions": [
                    "dashboard:read",
                    "api:read",
                    "audit:read"
                ],
                "level": 50
            },
            "viewer": {
                "name": "Viewer",
                "permissions": [
                    "dashboard:read"
                ],
                "level": 10
            }
        }

        logger.info("🔐 SecurityService inicializado")

    def _generate_jwt_secret(self) -> str:
        """Gera chave secreta para JWT."""
        return secrets.token_hex(32)

    def _generate_encryption_key(self) -> str:
        """Gera chave para encriptação Fernet."""
        # Usar PBKDF2 para derivar chave segura
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secrets.token_bytes(32)))
        return key.decode()

    # ============================================================================
    # AUTENTICAÇÃO
    # ============================================================================

    def hash_password(self, password: str) -> str:
        """Hash seguro de senha usando bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifica senha contra hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_access_token(self, user_id: int, username: str, roles: List[str]) -> str:
        """Cria JWT access token."""
        payload = {
            "sub": str(user_id),
            "username": username,
            "roles": roles,
            "type": "access",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.jwt_expiry
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def create_refresh_token(self, user_id: int) -> str:
        """Cria JWT refresh token."""
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.refresh_token_expiry
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def create_jwt_token(self, payload: Dict[str, Any]) -> str:
        """Cria JWT token com payload customizado."""
        token_payload = payload.copy()
        token_payload.update({
            "iat": datetime.utcnow(),
            "exp": token_payload.get("exp", datetime.utcnow() + self.jwt_expiry)
        })
        return jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verifica e decodifica JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            if payload.get("type") != token_type:
                return None

            # Verificar expiração
            from datetime import timezone
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if now > exp:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token inválido")
            return None

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica e decodifica JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token JWT expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token JWT inválido")
            return None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Autentica usuário.

        Returns
        -------
        dict or None
            Dados do usuário se autenticado, None caso contrário
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Buscar usuário
            cursor.execute("""
                SELECT id, username, password_hash, role, is_active, locked_until, login_attempts
                FROM users
                WHERE username = ?
            """, (username,))

            user = cursor.fetchone()
            if not user:
                self._log_audit("login_failed", None, f"Usuário não encontrado: {username}")
                return None

            user_id, db_username, password_hash, role, is_active, locked_until, login_attempts = user

            # Verificar se conta está ativa
            if not is_active:
                self._log_audit("login_failed", user_id, "Conta desativada")
                return None

            # Verificar se conta está bloqueada
            if locked_until and datetime.now() < datetime.fromisoformat(locked_until):
                self._log_audit("login_failed", user_id, "Conta bloqueada por tentativas excessivas")
                return None

            # Verificar senha
            if not self.verify_password(password, password_hash):
                # Incrementar tentativas de login
                new_attempts = login_attempts + 1
                if new_attempts >= self.max_login_attempts:
                    # Bloquear conta
                    lock_until = datetime.now() + self.lockout_duration
                    cursor.execute("""
                        UPDATE users
                        SET login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    """, (new_attempts, lock_until.isoformat(), user_id))
                    self._log_audit("account_locked", user_id, "Bloqueada por tentativas excessivas")
                else:
                    cursor.execute("""
                        UPDATE users
                        SET login_attempts = ?
                        WHERE id = ?
                    """, (new_attempts, user_id))

                conn.commit()
                self._log_audit("login_failed", user_id, f"Tentativa {new_attempts}/{self.max_login_attempts}")
                return None

            # Login bem-sucedido - resetar tentativas
            cursor.execute("""
                UPDATE users
                SET login_attempts = 0, locked_until = NULL, last_login = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), user_id))

            conn.commit()

            # Log de sucesso
            self._log_audit("login_success", user_id, "Login bem-sucedido")

            return {
                "id": user_id,
                "username": db_username,
                "role": role,
                "roles": [role],  # Para compatibilidade com RBAC
                "last_login": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro na autenticação: {e}")
            return None
        finally:
            conn.close()

    # ============================================================================
    # AUTORIZAÇÃO (RBAC)
    # ============================================================================

    def check_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """
        Verifica se usuário tem permissão específica.

        Parameters
        ----------
        user_roles : List[str]
            Roles do usuário
        required_permission : str
            Permissão requerida (ex: "dashboard:read")

        Returns
        -------
        bool
            True se tem permissão
        """
        for role in user_roles:
            if role in self.default_roles:
                permissions = self.default_roles[role]["permissions"]
                if required_permission in permissions:
                    return True

                # Verificar wildcards (ex: "dashboard:*" cobre "dashboard:read")
                for perm in permissions:
                    if perm.endswith("*"):
                        prefix = perm[:-1]
                        if required_permission.startswith(prefix):
                            return True

        return False

    def get_user_permissions(self, user_roles: List[str]) -> List[str]:
        """Retorna todas as permissões de um usuário."""
        permissions = set()

        for role in user_roles:
            if role in self.default_roles:
                permissions.update(self.default_roles[role]["permissions"])

        return list(permissions)

    def create_user(self, username: str, password: str, role: str = "viewer",
                   created_by: Optional[int] = None) -> bool:
        """
        Cria novo usuário.

        Parameters
        ----------
        username : str
            Nome do usuário
        password : str
            Senha em texto plano
        role : str
            Role do usuário
        created_by : int, optional
            ID do usuário que criou

        Returns
        -------
        bool
            True se criado com sucesso
        """
        if role not in self.default_roles:
            logger.error(f"Role inválida: {role}")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Verificar se usuário já existe
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                logger.error(f"Usuário já existe: {username}")
                return False

            # Criar usuário
            password_hash = self.hash_password(password)

            cursor.execute("""
                INSERT INTO users (username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 1, ?)
            """, (username, password_hash, role, datetime.now().isoformat()))

            user_id = cursor.lastrowid
            conn.commit()

            # Log de auditoria
            self._log_audit("user_created", user_id, f"Usuário criado por {created_by or 'system'}")

            return True

        except Exception as e:
            logger.error(f"Erro ao criar usuário: {e}")
            return False
        finally:
            conn.close()

    # ============================================================================
    # ENCRIPTAÇÃO
    # ============================================================================

    def encrypt_data(self, data: str) -> str:
        """Encripta dados sensíveis."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decripta dados sensíveis."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_config_value(self, key: str, value: Any) -> str:
        """Encripta valores de configuração sensíveis."""
        if key in ["api_key", "secret", "password", "token"]:
            return self.encrypt_data(json.dumps(value))
        return json.dumps(value)

    def decrypt_config_value(self, key: str, encrypted_value: str) -> Any:
        """Decripta valores de configuração."""
        if key in ["api_key", "secret", "password", "token"]:
            decrypted = self.decrypt_data(encrypted_value)
            return json.loads(decrypted)
        return json.loads(encrypted_value)

    # ============================================================================
    # AUDITORIA
    # ============================================================================

    def _log_audit(self, action: str, user_id: Optional[int], details: str,
                  resource: str = "auth", ip_address: Optional[str] = None):
        """Registra evento de auditoria."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO audit_log (user_id, action, resource, details, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, action, resource, details, ip_address, datetime.now().isoformat()))

            conn.commit()

        except Exception as e:
            logger.error(f"Erro ao registrar auditoria: {e}")
        finally:
            conn.close()

    def log_user_action(self, user_id: int, action: str, resource: str,
                       details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None):
        """Registra ação do usuário para auditoria."""
        details_str = json.dumps(details) if details else None
        self._log_audit(action, user_id, details_str or "", resource, ip_address)

    def get_audit_logs(self, user_id: Optional[int] = None, action: Optional[str] = None,
                      resource: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera logs de auditoria com filtros."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = """
                SELECT id, user_id, action, resource, details, ip_address, timestamp
                FROM audit_log
                WHERE 1=1
            """
            params = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if action:
                query += " AND action = ?"
                params.append(action)

            if resource:
                query += " AND resource = ?"
                params.append(resource)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            logs = []
            for row in cursor.fetchall():
                log_id, uid, act, res, det, ip, ts = row
                logs.append({
                    "id": log_id,
                    "user_id": uid,
                    "action": act,
                    "resource": res,
                    "details": json.loads(det) if det else None,
                    "ip_address": ip,
                    "timestamp": ts
                })

            return logs

        except Exception as e:
            logger.error(f"Erro ao buscar logs de auditoria: {e}")
            return []
        finally:
            conn.close()

    # ============================================================================
    # UTILITÁRIOS
    # ============================================================================

    def initialize_security_tables(self):
        """Inicializa tabelas de segurança no banco."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Tabela de usuários (se não existir)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'viewer',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)

            # Tabela de audit_log (se não existir)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # Tabela de config (se não existir)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    description TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_by TEXT
                )
            """)

            # Criar usuário admin padrão se não existir
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if cursor.fetchone()[0] == 0:
                admin_password = self.hash_password("admin123!")
                cursor.execute("""
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, 'admin', ?)
                """, ("admin", admin_password, datetime.now().isoformat()))

                logger.info("👤 Usuário admin criado (senha: admin123!)")

            conn.commit()
            logger.info("✅ Tabelas de segurança inicializadas")

        except Exception as e:
            logger.error(f"Erro ao inicializar tabelas de segurança: {e}")
        finally:
            conn.close()

    def get_security_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de segurança."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            stats = {}

            # Contagem de usuários por role
            cursor.execute("""
                SELECT role, COUNT(*) as count
                FROM users
                GROUP BY role
            """)
            stats["users_by_role"] = {row[0]: row[1] for row in cursor.fetchall()}

            # Tentativas de login recentes (últimas 24h)
            cursor.execute("""
                SELECT COUNT(*) as failed_logins
                FROM audit_log
                WHERE action = 'login_failed'
                AND timestamp >= datetime('now', '-1 day')
            """)
            stats["failed_logins_24h"] = cursor.fetchone()[0]

            # Contas bloqueadas
            cursor.execute("""
                SELECT COUNT(*) as locked_accounts
                FROM users
                WHERE locked_until > datetime('now')
            """)
            stats["locked_accounts"] = cursor.fetchone()[0]

            # Atividades recentes
            cursor.execute("""
                SELECT COUNT(*) as recent_activities
                FROM audit_log
                WHERE timestamp >= datetime('now', '-1 hour')
            """)
            stats["recent_activities"] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.error(f"Erro ao buscar estatísticas de segurança: {e}")
            return {}
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Busca usuário por ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, username, role, is_active, created_at, last_login
                FROM users
                WHERE id = ?
            """, (user_id,))

            user = cursor.fetchone()
            if not user:
                return None

            user_id, username, role, is_active, created_at, last_login = user

            return {
                "id": user_id,
                "username": username,
                "role": role,
                "is_active": bool(is_active),
                "created_at": created_at,
                "last_login": last_login,
                "permissions": self.get_user_permissions([role])
            }

        except Exception as e:
            logger.error(f"Erro ao buscar usuário: {e}")
            return None
        finally:
            conn.close()


# Singleton instance
_security_service = None

def get_security_service() -> SecurityService:
    """Factory function para serviço de segurança singleton."""
    global _security_service
    if _security_service is None:
        _security_service = SecurityService()
        _security_service.initialize_security_tables()
    return _security_service