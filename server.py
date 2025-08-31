import os
import re
import json
import time
import requests
import psycopg2
import pymysql
import sqlite3
import pymongo
from decimal import Decimal
from dotenv import load_dotenv
from google import genai
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import sqlalchemy
from sqlalchemy import create_engine, inspect, MetaData
from dataclasses import dataclass, asdict
import networkx as nx

load_dotenv()

# Data Models
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

# Abstract Database Adapter
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

# PostgreSQL Adapter
class PostgreSQLAdapter(DatabaseAdapter):
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

# MySQL Adapter
class MySQLAdapter(DatabaseAdapter):
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

# Enhanced Database Agent System
class EnhancedDatabaseAgentSystem:
    def __init__(self):
        # Database configuration
        self.db_configs = {
            'postgresql': {
                'host': os.getenv('POSTGRES_HOST'),
                'port': os.getenv('POSTGRES_PORT'),
                'user': os.getenv('POSTGRES_USER'),
                'password': os.getenv('POSTGRES_PASSWORD'),
                'database': os.getenv('POSTGRES_DB')
            },
            'mysql': {
                'host': os.getenv('MYSQL_HOST'),
                'port': os.getenv('MYSQL_PORT', 3306),
                'user': os.getenv('MYSQL_USER'),
                'password': os.getenv('MYSQL_PASSWORD'),
                'database': os.getenv('MYSQL_DB')
            }
        }
        
        # Initialize adapters
        self.adapters = {}
        self.current_adapter = None
        self.current_db_type = None
        
        # LLM configuration
        self.llm_client = genai.Client(api_key=os.getenv("GEMINI_API"))
        self.model_config = {
            'primary': 'gemini-2.0-flash-exp',
            'fallback': 'gemini-1.5-flash'
        }
        
        # Schema cache
        self.schema_cache = {}
        self.schema_cache_ttl = 3600  # 1 hour
        
        # Context management
        self.context_history = []
        self.max_context_length = 10
        
        # Query optimization
        self.query_optimizer = QueryOptimizer()
        
        # Relationship graph
        self.relationship_graph = nx.DiGraph()
        
        # Initialize notification channels
        self.notification_channels = {
            'telegram': {
                'token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID')
            }
        }
    
    def send_telegram_notification(self, message: str):
        """Send notification to Telegram"""
        if 'telegram' in self.notification_channels:
            tg = self.notification_channels['telegram']
            if tg['token'] and tg['chat_id']:
                url = f"https://api.telegram.org/bot{tg['token']}/sendMessage"
                params = {
                    'chat_id': tg['chat_id'],
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                try:
                    requests.get(url, params=params)
                except:
                    pass
    
    def connect_database(self, db_type: str = 'postgresql') -> bool:
        """Connect to specified database type"""
        try:
            if db_type not in self.db_configs:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            config = self.db_configs[db_type]
            
            # Check if all required config values exist
            if not all(config.values()):
                print(f"Missing configuration for {db_type}")
                return False
            
            # Create appropriate adapter
            if db_type == 'postgresql':
                adapter = PostgreSQLAdapter(config)
            elif db_type == 'mysql':
                adapter = MySQLAdapter(config)
            else:
                raise ValueError(f"Adapter not implemented for {db_type}")
            
            # Connect
            adapter.connect()
            self.adapters[db_type] = adapter
            self.current_adapter = adapter
            self.current_db_type = db_type
            
            print(f"✅ Connected to {db_type} database")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to {db_type}: {e}")
            return False
    
    def analyze_database_smart(self) -> DatabaseSchema:
        """Smart database analysis with caching and relationship detection"""
        if not self.current_adapter:
            raise RuntimeError("No database connected")
        
        # Check cache
        cache_key = f"{self.current_db_type}_schema"
        if cache_key in self.schema_cache:
            cached_time, cached_schema = self.schema_cache[cache_key]
            if time.time() - cached_time < self.schema_cache_ttl:
                print("📦 Using cached schema")
                return cached_schema
        
        print("🔍 Analyzing database schema...")
        schema = self.current_adapter.analyze_schema()
        
        # Build relationship graph
        self._build_relationship_graph(schema)
        
        # Detect implicit relationships
        implicit_relationships = self._detect_implicit_relationships(schema)
        schema.relationships.extend(implicit_relationships)
        
        # Cache the schema
        self.schema_cache[cache_key] = (time.time(), schema)
        
        return schema
    
    def _build_relationship_graph(self, schema: DatabaseSchema):
        """Build a graph of table relationships"""
        self.relationship_graph.clear()
        
        # Add nodes (tables)
        for table_name in schema.tables:
            self.relationship_graph.add_node(table_name)
        
        # Add edges (relationships)
        for rel in schema.relationships:
            self.relationship_graph.add_edge(
                rel['from_table'],
                rel['to_table'],
                **rel
            )
    
    def _detect_implicit_relationships(self, schema: DatabaseSchema) -> List[Dict[str, Any]]:
        """Detect implicit relationships based on naming conventions"""
        implicit_rels = []
        
        for table1_name, table1 in schema.tables.items():
            for table2_name, table2 in schema.tables.items():
                if table1_name == table2_name:
                    continue
                
                # Check for common naming patterns
                for col1 in table1.columns:
                    col1_name = col1['name'].lower()
                    
                    # Pattern 1: table2_id in table1
                    if col1_name == f"{table2_name.lower()}_id" or col1_name == f"{table2_name.lower()}id":
                        # Check if this is already a known FK
                        is_known = any(
                            fk['column'] == col1['name'] and fk['referenced_table'] == table2_name
                            for fk in table1.foreign_keys
                        )
                        
                        if not is_known:
                            implicit_rels.append({
                                'from_table': table1_name,
                                'from_column': col1['name'],
                                'to_table': table2_name,
                                'to_column': 'id',  # Assume 'id' is primary key
                                'type': 'implicit_foreign_key',
                                'confidence': 0.8
                            })
                    
                    # Pattern 2: Common column names suggesting relationship
                    for col2 in table2.columns:
                        if col1['name'] == col2['name'] and col1['type'] == col2['type']:
                            if col1['name'] in ['created_by', 'updated_by', 'user_id', 'owner_id']:
                                implicit_rels.append({
                                    'from_table': table1_name,
                                    'from_column': col1['name'],
                                    'to_table': table2_name,
                                    'to_column': col2['name'],
                                    'type': 'implicit_reference',
                                    'confidence': 0.6
                                })
        
        return implicit_rels
    
    def enhanced_context_agent(self, question: str, schema: DatabaseSchema) -> Dict[str, Any]:
        """Enhanced context analysis using schema information"""
        context = {
            'language': self._detect_language(question),
            'complexity': 'simple',
            'operation_type': 'read',
            'mentioned_tables': [],
            'mentioned_columns': [],
            'requires_join': False,
            'requires_aggregation': False,
            'time_based': False,
            'limit_specified': False,
            'order_by_needed': False
        }
        
        question_lower = question.lower()
        
        # Detect mentioned tables dynamically
        for table_name in schema.tables.keys():
            table_lower = table_name.lower()
            # Check for table name or common variations
            if table_lower in question_lower or table_lower.rstrip('s') in question_lower:
                context['mentioned_tables'].append(table_name)
        
        # Detect mentioned columns
        for table_name, table_info in schema.tables.items():
            for col in table_info.columns:
                col_lower = col['name'].lower()
                if col_lower in question_lower:
                    context['mentioned_columns'].append({
                        'table': table_name,
                        'column': col['name'],
                        'type': col['type']
                    })
        
        # Detect operation type
        write_keywords = ['insert', 'add', 'create', 'update', 'modify', 'delete', 'remove', 'drop']
        if any(kw in question_lower for kw in write_keywords):
            context['operation_type'] = 'write'
        
        # Detect aggregation needs
        agg_keywords = ['count', 'sum', 'average', 'avg', 'max', 'min', 'total', 'group by']
        if any(kw in question_lower for kw in agg_keywords):
            context['requires_aggregation'] = True
        
        # Detect join requirements
        if len(context['mentioned_tables']) > 1:
            context['requires_join'] = True
        elif any(kw in question_lower for kw in ['related', 'associated', 'connected', 'with their', 'including']):
            context['requires_join'] = True
        
        # Detect ordering needs
        order_keywords = ['top', 'first', 'last', 'highest', 'lowest', 'best', 'worst', 'latest', 'earliest']
        if any(kw in question_lower for kw in order_keywords):
            context['order_by_needed'] = True
        
        # Detect limit
        limit_patterns = [r'\btop\s+(\d+)', r'\bfirst\s+(\d+)', r'\blast\s+(\d+)', r'\blimit\s+(\d+)']
        for pattern in limit_patterns:
            match = re.search(pattern, question_lower)
            if match:
                context['limit_specified'] = True
                context['limit_value'] = int(match.group(1))
                break
        
        # Detect time-based queries
        time_keywords = ['today', 'yesterday', 'this week', 'last week', 'this month', 'last month', 
                        'this year', 'last year', 'between', 'since', 'before', 'after']
        if any(kw in question_lower for kw in time_keywords):
            context['time_based'] = True
        
        # Determine complexity
        complexity_score = sum([
            context['requires_join'] * 2,
            context['requires_aggregation'] * 2,
            context['time_based'] * 1,
            len(context['mentioned_tables']) > 2,
            len(context['mentioned_columns']) > 3
        ])
        
        if complexity_score >= 3:
            context['complexity'] = 'complex'
        elif complexity_score >= 1:
            context['complexity'] = 'medium'
        
        return context
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Vietnamese or English"""
        vietnamese_chars = 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
        return 'vi' if any(char in text.lower() for char in vietnamese_chars) else 'en'
    
    def intelligent_sql_generation(self, question: str, context: Dict, schema: DatabaseSchema) -> str:
        """Generate SQL using schema understanding and context"""
        
        # Create detailed schema description
        schema_description = self._create_schema_description(schema, context)
        
        # Build relationship context
        relationship_context = self._build_relationship_context(context['mentioned_tables'], schema)
        
        prompt = f"""
        You are an expert SQL developer. Generate an optimal SQL query based on the following:
        
        Database Type: {self.current_db_type}
        
        User Question: {question}
        
        Context Analysis:
        {json.dumps(context, indent=2)}
        
        Relevant Schema:
        {schema_description}
        
        Table Relationships:
        {relationship_context}
        
        Rules:
        1. Use appropriate SQL dialect for {self.current_db_type}
        2. Only use tables and columns that exist in the schema
        3. Handle NULL values appropriately
        4. Use proper JOIN types based on relationships
        5. Apply appropriate WHERE clauses for filtering
        6. Use window functions for ranking queries if needed
        7. Add LIMIT clause if not specified but seems needed
        8. Optimize for performance using available indexes
        9. For write operations (INSERT, UPDATE, DELETE) that may affect related tables or require order due to constraints, generate multiple SQL statements if necessary, separated by ';'. Ensure the order respects foreign key dependencies (parent tables first for inserts).
        10. If the query involves inserting data into multiple related tables, use a single SQL statement with Common Table Expressions (CTE) to insert into parent tables first and retrieve IDs for child tables to avoid constraint violations and ensure atomicity. Example:
           WITH new_parent AS (
             INSERT INTO parent (...) VALUES (...) RETURNING id
           )
           INSERT INTO child (parent_id, ...) SELECT id, ... FROM new_parent;
        
        Generate ONLY the SQL query, no explanations:
        ```sql
        """
        
        try:
            response = self.llm_client.models.generate_content(
                model=self.model_config['primary'],
                contents=prompt
            )
            
            # Extract SQL
            sql_match = re.search(r'```sql\n(.*?)\n```', response.text, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip()
            else:
                sql = response.text.strip()
            
            # Validate and optimize the SQL
            sql = self.query_optimizer.optimize(sql, schema, self.current_db_type)
            
            return sql
            
        except Exception as e:
            print(f"SQL generation error: {e}")
            # Fallback to simple query generation
            return self._generate_fallback_sql(question, context, schema)
    
    def _create_schema_description(self, schema: DatabaseSchema, context: Dict) -> str:
        """Create focused schema description based on context"""
        description = []
        
        # If specific tables mentioned, focus on those
        tables_to_describe = context['mentioned_tables'] if context['mentioned_tables'] else list(schema.tables.keys())[:5]
        
        for table_name in tables_to_describe:
            if table_name not in schema.tables:
                continue
                
            table_info = schema.tables[table_name]
            desc = f"\nTable: {table_name} ({table_info.row_count} rows)\n"
            desc += "Columns:\n"
            
            for col in table_info.columns[:10]:  # Limit to first 10 columns
                desc += f"  - {col['name']}: {col['type']}"
                if not col['nullable']:
                    desc += " NOT NULL"
                if col['name'] in table_info.primary_keys:
                    desc += " PRIMARY KEY"
                desc += "\n"
            
            if table_info.foreign_keys:
                desc += "Foreign Keys:\n"
                for fk in table_info.foreign_keys:
                    desc += f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}\n"
            
            description.append(desc)
        
        return "\n".join(description)
    
    def _build_relationship_context(self, tables: List[str], schema: DatabaseSchema) -> str:
        """Build context about relationships between mentioned tables"""
        if len(tables) < 2:
            return "No multi-table relationships needed"
        
        relationships = []
        
        for rel in schema.relationships:
            if rel['from_table'] in tables and rel['to_table'] in tables:
                rel_desc = f"{rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}"
                if rel.get('type') == 'implicit_foreign_key':
                    rel_desc += " (implicit)"
                relationships.append(rel_desc)
        
        if relationships:
            return "Join paths:\n" + "\n".join(f"  - {r}" for r in relationships)
        
        # Try to find indirect paths
        if self.relationship_graph.number_of_nodes() > 0:
            for t1 in tables:
                for t2 in tables:
                    if t1 != t2 and self.relationship_graph.has_node(t1) and self.relationship_graph.has_node(t2):
                        try:
                            path = nx.shortest_path(self.relationship_graph, t1, t2)
                            if len(path) > 2:
                                relationships.append(f"Indirect: {' -> '.join(path)}")
                        except nx.NetworkXNoPath:
                            pass
        
        return "Join paths:\n" + "\n".join(f"  - {r}" for r in relationships) if relationships else "No direct relationships found"
    
    def _generate_fallback_sql(self, question: str, context: Dict, schema: DatabaseSchema) -> str:
        """Generate fallback SQL when main generation fails"""
        sql_parts = []
        
        # SELECT clause
        if context['requires_aggregation']:
            sql_parts.append("SELECT COUNT(*) as count")
        else:
            sql_parts.append("SELECT *")
        
        # FROM clause
        if context['mentioned_tables']:
            sql_parts.append(f"FROM {context['mentioned_tables'][0]}")
        else:
            # Use first table in schema
            sql_parts.append(f"FROM {list(schema.tables.keys())[0]}")
        
        # Add LIMIT
        if context.get('limit_value'):
            sql_parts.append(f"LIMIT {context['limit_value']}")
        else:
            sql_parts.append("LIMIT 100")
        
        return " ".join(sql_parts)
    
    def process_query(self, question: str) -> str:
        """Main query processing pipeline"""
        print(f"\n🔍 Processing: {question}")
        start_time = time.time()
        
        try:
            # Ensure database connection
            if not self.current_adapter:
                return "❌ No database connected. Please connect first."
            
            # Analyze schema
            schema = self.analyze_database_smart()
            
            # Context analysis with schema awareness
            context = self.enhanced_context_agent(question, schema)
            print(f"📊 Context: {context['complexity']} query, {len(context['mentioned_tables'])} tables")
            
            # Generate SQL
            sql = self.intelligent_sql_generation(question, context, schema)
            print(f"✨ Generated SQL: {sql[:100]}...")
            
            # Validate SQL
            is_valid, error = self.current_adapter.validate_sql(sql)
            if not is_valid:
                print(f"⚠️ SQL validation failed: {error}")
                # Try to fix and regenerate
                sql = self._fix_sql_errors(sql, error, schema)
            
            # Execute query
            result, error = self.current_adapter.execute_query(sql)
            
            if error:
                return f"❌ Query execution failed: {error}"
            
            # Format response
            response = self._format_response(result, question, context)
            
            # Send Telegram report
            self.send_telegram_notification(f"**Query Processed**\nQuestion: {question}\nSQL:\n```sql\n{sql}\n```\nResponse:\n{response}")
            
            # Update context history
            self.context_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'sql': sql,
                'context': context,
                'execution_time': time.time() - start_time,
                'success': True
            })
            
            # Trim history
            if len(self.context_history) > self.max_context_length:
                self.context_history = self.context_history[-self.max_context_length:]
            
            print(f"✅ Completed in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _fix_sql_errors(self, sql: str, error: str, schema: DatabaseSchema) -> str:
        """Attempt to fix SQL errors using LLM"""
        prompt = f"""
        Fix the following SQL query error:
        
        Original SQL:
        {sql}
        
        Error:
        {error}
        
        Database Type: {self.current_db_type}
        
        Available Tables:
        {', '.join(schema.tables.keys())}
        
        Provide only the corrected SQL:
        ```sql
        """
        
        try:
            response = self.llm_client.models.generate_content(
                model=self.model_config['primary'],
                contents=prompt
            )
            
            sql_match = re.search(r'```sql\n(.*?)\n```', response.text, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            return sql
        except:
            return sql
    
    def _format_response(self, result: Dict, question: str, context: Dict) -> str:
        """Format query results into readable response"""
        if 'rows' in result:
            response = f"Found {result['row_count']} results:\n\n"
            
            if result['row_count'] > 0:
                # Create table format
                columns = result['columns']
                rows = result['rows'][:10]  # Show first 10 rows
                
                # Calculate column widths
                widths = [len(str(col)) for col in columns]
                for row in rows:
                    for i, val in enumerate(row):
                        widths[i] = max(widths[i], len(str(val)))
                
                # Header
                header = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
                separator = "-+-".join("-" * w for w in widths)
                
                response += header + "\n"
                response += separator + "\n"
                
                # Rows
                for row in rows:
                    response += " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row)) + "\n"
                
                if result['row_count'] > 10:
                    response += f"\n... and {result['row_count'] - 10} more rows"
            
        elif 'affected_rows' in result:
            response = f"Query executed successfully. {result['affected_rows']} rows affected."
        else:
            response = "Query executed successfully."
        
        return response


# Query Optimizer Class
class QueryOptimizer:
    """Optimize SQL queries for better performance"""
    
    def optimize(self, sql: str, schema: DatabaseSchema, db_type: str) -> str:
        """Optimize SQL query based on schema and database type"""
        optimized = sql
        
        # Remove unnecessary SELECT *
        optimized = self._optimize_select_clause(optimized, schema)
        
        # Add appropriate indexes hints if needed
        optimized = self._add_index_hints(optimized, schema, db_type)
        
        # Optimize JOIN order
        optimized = self._optimize_join_order(optimized, schema)
        
        return optimized
    
    def _optimize_select_clause(self, sql: str, schema: DatabaseSchema) -> str:
        """Optimize SELECT clause to avoid SELECT *"""
        # This is a simplified implementation
        # In production, use proper SQL parser
        return sql
    
    def _add_index_hints(self, sql: str, schema: DatabaseSchema, db_type: str) -> str:
        """Add index hints for better performance"""
        # Database-specific index hints
        return sql
    
    def _optimize_join_order(self, sql: str, schema: DatabaseSchema) -> str:
        """Optimize JOIN order based on table sizes"""
        # Reorder JOINs from smallest to largest table
        return sql


# CLI Interface
def main():
    """Enhanced CLI interface"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 Multi-Database Intelligent Agent System 🤖        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    agent = EnhancedDatabaseAgentSystem()
    
    # Database selection
    print("\n📊 Available Database Types:")
    print("1. PostgreSQL")
    print("2. MySQL")
    print("3. SQLite (coming soon)")
    print("4. MongoDB (coming soon)")
    
    db_choice = input("\nSelect database type (1-2): ").strip()
    
    db_type_map = {
        '1': 'postgresql',
        '2': 'mysql'
    }
    
    selected_db = db_type_map.get(db_choice, 'postgresql')
    
    # Connect to database
    print(f"\n🔌 Connecting to {selected_db}...")
    if not agent.connect_database(selected_db):
        print("Failed to connect to database. Please check your configuration.")
        return
    
    # Analyze schema
    print("\n🔍 Analyzing database schema...")
    schema = agent.analyze_database_smart()
    
    print(f"\n✅ Connected to {selected_db} database")
    print(f"📊 Found {len(schema.tables)} tables")
    print(f"🔗 Found {len(schema.relationships)} relationships")
    
    # Show available tables
    print("\n📋 Available tables:")
    for i, table_name in enumerate(list(schema.tables.keys())[:10], 1):
        table = schema.tables[table_name]
        print(f"  {i}. {table_name} ({table.row_count} rows, {len(table.columns)} columns)")
    
    if len(schema.tables) > 10:
        print(f"  ... and {len(schema.tables) - 10} more tables")
    
    print("\n" + "="*60)
    print("💡 Commands:")
    print("  - Type your question in natural language")
    print("  - 'SCHEMA <table>' - Show table schema")
    print("  - 'RELATIONSHIPS' - Show table relationships")
    print("  - 'SWITCH DB' - Switch database type")
    print("  - 'CLEAR' - Clear context history")
    print("  - 'EXIT' - Exit the system")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.upper() == 'EXIT':
                print("\n👋 Goodbye!")
                break
            
            elif user_input.upper().startswith('SCHEMA'):
                parts = user_input.split()
                if len(parts) > 1:
                    table_name = parts[1]
                    if table_name in schema.tables:
                        table = schema.tables[table_name]
                        print(f"\n📋 Schema for {table_name}:")
                        print(f"Rows: {table.row_count}")
                        print("\nColumns:")
                        for col in table.columns:
                            print(f"  - {col['name']}: {col['type']} {'NOT NULL' if not col['nullable'] else 'NULL'}")
                        if table.foreign_keys:
                            print("\nForeign Keys:")
                            for fk in table.foreign_keys:
                                print(f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
                    else:
                        print(f"Table '{table_name}' not found")
                else:
                    print("Usage: SCHEMA <table_name>")
            
            elif user_input.upper() == 'RELATIONSHIPS':
                print("\n🔗 Table Relationships:")
                for rel in schema.relationships[:20]:
                    rel_type = rel.get('type', 'foreign_key')
                    confidence = f" (confidence: {rel.get('confidence', 1.0):.1%})" if 'confidence' in rel else ""
                    print(f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']} [{rel_type}]{confidence}")
                if len(schema.relationships) > 20:
                    print(f"  ... and {len(schema.relationships) - 20} more relationships")
            
            elif user_input.upper() == 'SWITCH DB':
                print("\nSelect new database type:")
                print("1. PostgreSQL")
                print("2. MySQL")
                db_choice = input("Choice (1-2): ").strip()
                new_db = db_type_map.get(db_choice, 'postgresql')
                if agent.connect_database(new_db):
                    schema = agent.analyze_database_smart()
                    print(f"✅ Switched to {new_db}")
            
            elif user_input.upper() == 'CLEAR':
                agent.context_history.clear()
                agent.schema_cache.clear()
                print("🗑️ Context and cache cleared")
            
            else:
                # Process natural language query
                response = agent.process_query(user_input)
                print("\n" + "="*60)
                print(response)
                print("="*60)
        
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'EXIT' to quit")


if __name__ == "__main__":
    main()