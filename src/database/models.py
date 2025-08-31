"""
Data models for database schema representation
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TableInfo:
    """Information about a database table"""
    name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[Dict[str, Any]]
    row_count: int
    sample_data: List[Any]
    description: Optional[str] = None


@dataclass
class DatabaseSchema:
    """Complete database schema information"""
    db_type: str
    tables: Dict[str, TableInfo]
    relationships: List[Dict[str, Any]]
    views: List[str]
    stored_procedures: List[str]
    metadata: Dict[str, Any]
