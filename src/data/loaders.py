# src/data/loaders.py
"""
Concrete data loader implementations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import sqlite3
import json

from .base import DataLoader
from ..utils import logger
from ..config import settings


class CSVLoader(DataLoader):
    """CSV data loader with validation and preprocessing"""

    def __init__(self, name: str = "csv_loader", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.file_path = None
        self.encoding = self.config.get('encoding', 'utf-8')
        self.separator = self.config.get('separator', ',')
        self.has_header = self.config.get('has_header', True)
        self.date_columns = self.config.get('date_columns', [])
        self.dtype_mapping = self.config.get('dtype_mapping', {})

    def load_data(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            source: Path to CSV file
            **kwargs: Additional pandas.read_csv parameters

        Returns:
            Loaded DataFrame
        """
        self.file_path = Path(source)

        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.logger.info(f"Loading CSV data from {self.file_path}")

        # Prepare read_csv parameters
        read_params = {
            'encoding': self.encoding,
            'sep': self.separator,
            'header': 0 if self.has_header else None,
            'low_memory': False,  # Avoid dtype warnings
            **kwargs
        }

        try:
            # Load data
            df = pd.read_csv(self.file_path, **read_params)

            # Apply dtype mapping if specified
            if self.dtype_mapping:
                df = df.astype(self.dtype_mapping)

            # Parse date columns
            if self.date_columns:
                for date_col in self.date_columns:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from CSV")
            return df

        except Exception as e:
            error_msg = f"Failed to load CSV file {self.file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def validate_input(self, data: Any) -> bool:
        """Validate input for CSV loader"""
        if isinstance(data, (str, Path)):
            path = Path(data)
            return path.exists() and path.is_file() and path.suffix.lower() == '.csv'
        return False


class DatabaseLoader(DataLoader):
    """Database loader for SQL databases"""

    def __init__(self, name: str = "db_loader", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.connection_string = self.config.get('connection_string')
        self.table_name = self.config.get('table_name')
        self.query = self.config.get('query')
        self.chunk_size = self.config.get('chunk_size', 10000)

    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from database

        Args:
            source: Connection string or table name
            **kwargs: Additional parameters

        Returns:
            Loaded DataFrame
        """
        if not self.connection_string and not source:
            raise ValueError("Connection string required for database loading")

        conn_str = self.connection_string or source
        table = self.table_name or source

        self.logger.info(f"Loading data from database table: {table}")

        try:
            # For SQLite (common in AML projects)
            if conn_str.endswith('.db') or 'sqlite' in conn_str:
                conn = sqlite3.connect(conn_str)
            else:
                raise NotImplementedError("Only SQLite databases supported currently")

            # Build query
            if self.query:
                query = self.query
            else:
                query = f"SELECT * FROM {table}"

            # Load data
            if self.chunk_size and self.chunk_size > 0:
                # Load in chunks for large tables
                chunks = []
                for chunk in pd.read_sql_query(query, conn, chunksize=self.chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql_query(query, conn)

            conn.close()

            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from database")
            return df

        except Exception as e:
            error_msg = f"Failed to load data from database: {str(e)}"
            self.logger.error(error_msg)
            raise

    def validate_input(self, data: Any) -> bool:
        """Validate input for database loader"""
        # Database loader accepts connection strings or table names
        return isinstance(data, str) and len(data.strip()) > 0


class JSONLoader(DataLoader):
    """JSON data loader"""

    def __init__(self, name: str = "json_loader", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.orient = self.config.get('orient', 'records')  # 'records', 'split', 'index', etc.

    def load_data(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file

        Args:
            source: Path to JSON file
            **kwargs: Additional parameters

        Returns:
            Loaded DataFrame
        """
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        self.logger.info(f"Loading JSON data from {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Handle different JSON orientations
            if self.orient == 'records':
                # Data is already in records format
                pass
            elif self.orient == 'split':
                # Data has separate index, columns, data
                if isinstance(data, dict) and 'data' in data:
                    df = pd.DataFrame(data['data'], columns=data.get('columns'), index=data.get('index'))

            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from JSON")
            return df

        except Exception as e:
            error_msg = f"Failed to load JSON file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def validate_input(self, data: Any) -> bool:
        """Validate input for JSON loader"""
        if isinstance(data, (str, Path)):
            path = Path(data)
            return path.exists() and path.is_file() and path.suffix.lower() == '.json'
        return False


class ParquetLoader(DataLoader):
    """Parquet data loader for efficient storage"""

    def __init__(self, name: str = "parquet_loader", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.engine = self.config.get('engine', 'auto')  # 'auto', 'pyarrow', 'fastparquet'

    def load_data(self, source: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet file

        Args:
            source: Path to Parquet file
            **kwargs: Additional parameters

        Returns:
            Loaded DataFrame
        """
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        self.logger.info(f"Loading Parquet data from {file_path}")

        try:
            # Load Parquet file
            df = pd.read_parquet(file_path, engine=self.engine, **kwargs)

            self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from Parquet")
            return df

        except Exception as e:
            error_msg = f"Failed to load Parquet file {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def validate_input(self, data: Any) -> bool:
        """Validate input for Parquet loader"""
        if isinstance(data, (str, Path)):
            path = Path(data)
            return path.exists() and path.is_file() and path.suffix.lower() in ['.parquet', '.pq']
        return False


__all__ = [
    "CSVLoader",
    "DatabaseLoader",
    "JSONLoader",
    "ParquetLoader"
]