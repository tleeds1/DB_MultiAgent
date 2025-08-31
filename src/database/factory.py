"""
Database factory for creating appropriate database adapters
"""

from typing import Dict, Any
from .adapters import DatabaseAdapter, PostgreSQLAdapter, MySQLAdapter


class DatabaseFactory:
    """Factory class to create appropriate database connector"""
    
    @staticmethod
    def create_connector(db_type: str, config: Dict[str, Any]) -> DatabaseAdapter:
        """Create database adapter based on type"""
        if db_type.lower() in ['postgresql', 'postgres']:
            return PostgreSQLAdapter(config)
        elif db_type.lower() == 'mysql':
            return MySQLAdapter(config)
        elif db_type.lower() == 'sqlite':
            # TODO: Implement SQLite adapter
            raise NotImplementedError("SQLite adapter not yet implemented")
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @staticmethod
    def get_supported_types() -> list:
        """Get list of supported database types"""
        return ['postgresql', 'mysql']  # Add more as implemented
    
    @staticmethod
    def get_required_config(db_type: str) -> list:
        """Get required configuration keys for database type"""
        configs = {
            'postgresql': ['host', 'port', 'user', 'password', 'database'],
            'mysql': ['host', 'user', 'password', 'database'],
            'sqlite': ['database']
        }
        return configs.get(db_type.lower(), [])
