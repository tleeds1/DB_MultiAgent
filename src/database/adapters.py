"""
Database adapters for different database types
"""

import psycopg2
import pymysql
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine, inspect, MetaData
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from .models import DatabaseSchema, TableInfo


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> Any:
        """Connect to database"""
        pass
    
    @abstractmethod
    def analyze_schema(self) -> DatabaseSchema:
        """Analyze database schema"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute a query"""
        pass
    
    @abstractmethod
    def get_table_relationships(self) -> List[Dict[str, Any]]:
        """Get relationships between tables"""
        pass
    
    @abstractmethod
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax for this database"""
        pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.engine = None
        
    def connect(self, config: Dict[str, Any] = None) -> Any:
        """Connect to PostgreSQL database"""
        if config:
            self.config = config
        
        try:
            # Using SQLAlchemy for better metadata inspection
            conn_string = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            self.engine = create_engine(conn_string)
            self.connection = psycopg2.connect(**self.config)
            return self.connection
        except Exception as e:
            raise ConnectionError(f"PostgreSQL connection failed: {e}")
    
    def analyze_schema(self) -> DatabaseSchema:
        """Comprehensive PostgreSQL schema analysis"""
        if not self.engine:
            self.connect()
        
        inspector = inspect(self.engine)
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        
        tables = {}
        relationships = []
        
        # Analyze each table
        for table_name in inspector.get_table_names():
            # Get columns with detailed info
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'default': col['default'],
                    'autoincrement': col.get('autoincrement', False),
                    'comment': col.get('comment', '')
                })
            
            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_keys = pk_constraint['constrained_columns'] if pk_constraint else []
            
            # Get foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'column': fk['constrained_columns'][0] if fk['constrained_columns'] else None,
                    'referenced_table': fk['referred_table'],
                    'referenced_column': fk['referred_columns'][0] if fk['referred_columns'] else None
                })
                
                # Add to relationships
                relationships.append({
                    'from_table': table_name,
                    'from_column': fk['constrained_columns'][0] if fk['constrained_columns'] else None,
                    'to_table': fk['referred_table'],
                    'to_column': fk['referred_columns'][0] if fk['referred_columns'] else None,
                    'type': 'foreign_key'
                })
            
            # Get indexes
            indexes = []
            for idx in inspector.get_indexes(table_name):
                indexes.append({
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx['unique']
                })
            
            # Get row count and sample data
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                sample_data = cursor.fetchall()
            
            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=row_count,
                sample_data=sample_data
            )
        
        # Get views
        views = inspector.get_view_names()
        
        # Get stored procedures (PostgreSQL functions)
        stored_procedures = []
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT proname FROM pg_proc 
                WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """)
            stored_procedures = [row[0] for row in cursor.fetchall()]
        
        return DatabaseSchema(
            db_type='postgresql',
            tables=tables,
            relationships=relationships,
            views=views,
            stored_procedures=stored_procedures,
            metadata={
                'version': self._get_db_version(),
                'size': self._get_db_size()
            }
        )
    
    def execute_query(self, query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute PostgreSQL query"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                self.connection.commit()
                return {
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows)
                }, None
            else:
                self.connection.commit()
                return {
                    'affected_rows': cursor.rowcount
                }, None
                
        except Exception as e:
            self.connection.rollback()
            return None, str(e)
    
    def get_table_relationships(self) -> List[Dict[str, Any]]:
        """Get detailed table relationships"""
        relationships = []
        inspector = inspect(self.engine)
        
        for table in inspector.get_table_names():
            for fk in inspector.get_foreign_keys(table):
                relationships.append({
                    'from_table': table,
                    'from_columns': fk['constrained_columns'],
                    'to_table': fk['referred_table'],
                    'to_columns': fk['referred_columns'],
                    'constraint_name': fk['name']
                })
        
        return relationships
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate PostgreSQL SQL syntax"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _get_db_version(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute("SELECT version()")
        return cursor.fetchone()[0]
    
    def _get_db_size(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT pg_size_pretty(pg_database_size('{self.config['database']}'))")
        return cursor.fetchone()[0]


class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.engine = None
    
    def connect(self, config: Dict[str, Any] = None) -> Any:
        """Connect to MySQL database"""
        if config:
            self.config = config
        
        try:
            conn_string = f"mysql+pymysql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config.get('port', 3306)}/{self.config['database']}"
            self.engine = create_engine(conn_string)
            self.connection = pymysql.connect(
                host=self.config['host'],
                port=int(self.config.get('port', 3306)),
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            return self.connection
        except Exception as e:
            raise ConnectionError(f"MySQL connection failed: {e}")
    
    def analyze_schema(self) -> DatabaseSchema:
        """Comprehensive MySQL schema analysis"""
        if not self.engine:
            self.connect()
        
        inspector = inspect(self.engine)
        tables = {}
        relationships = []
        
        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable'],
                    'default': col['default'],
                    'autoincrement': col.get('autoincrement', False)
                })
            
            pk_constraint = inspector.get_pk_constraint(table_name)
            primary_keys = pk_constraint['constrained_columns'] if pk_constraint else []
            
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'column': fk['constrained_columns'][0],
                    'referenced_table': fk['referred_table'],
                    'referenced_column': fk['referred_columns'][0]
                })
                
                relationships.append({
                    'from_table': table_name,
                    'from_column': fk['constrained_columns'][0],
                    'to_table': fk['referred_table'],
                    'to_column': fk['referred_columns'][0],
                    'type': 'foreign_key'
                })
            
            indexes = []
            for idx in inspector.get_indexes(table_name):
                indexes.append({
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx['unique']
                })
            
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                sample_data = cursor.fetchall()
            
            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                indexes=indexes,
                row_count=row_count,
                sample_data=sample_data
            )
        
        views = inspector.get_view_names()
        
        # Get stored procedures
        stored_procedures = []
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW PROCEDURE STATUS WHERE Db = %s", (self.config['database'],))
            stored_procedures = [row[1] for row in cursor.fetchall()]
        
        return DatabaseSchema(
            db_type='mysql',
            tables=tables,
            relationships=relationships,
            views=views,
            stored_procedures=stored_procedures,
            metadata={
                'version': self._get_db_version(),
                'size': self._get_db_size()
            }
        )
    
    def execute_query(self, query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute MySQL query"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                self.connection.commit()
                return {
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows)
                }, None
            else:
                self.connection.commit()
                return {
                    'affected_rows': cursor.rowcount
                }, None
                
        except Exception as e:
            self.connection.rollback()
            return None, str(e)
    
    def get_table_relationships(self) -> List[Dict[str, Any]]:
        """Get detailed table relationships"""
        relationships = []
        inspector = inspect(self.engine)
        
        for table in inspector.get_table_names():
            for fk in inspector.get_foreign_keys(table):
                relationships.append({
                    'from_table': table,
                    'from_columns': fk['constrained_columns'],
                    'to_table': fk['referred_table'],
                    'to_columns': fk['referred_columns'],
                    'constraint_name': fk['name']
                })
        
        return relationships
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate MySQL SQL syntax"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _get_db_version(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute("SELECT VERSION()")
        return cursor.fetchone()[0]
    
    def _get_db_size(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'DB Size in MB'
            FROM information_schema.TABLES
            WHERE table_schema = '{self.config['database']}'
        """)
        size = cursor.fetchone()[0]
        return f"{size} MB"
