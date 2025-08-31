"""
Main enhanced database agent system
"""

import os
import time
import json
import re
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from google import genai

from ..database import DatabaseFactory, DatabaseAdapter
from ..database.models import DatabaseSchema
from ..utils.query_optimizer import QueryOptimizer
from ..utils.notification import NotificationManager
from ..utils.schema_analyzer import SchemaAnalyzer
from .context_agent import ContextAgent
from .sql_generation_agent import SQLGenerationAgent
from .execution_agent import ExecutionAgent

# Load environment variables
load_dotenv()


class EnhancedDatabaseAgentSystem:
    """Enhanced database agent system with multi-database support"""
    
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
                'port': int(os.getenv('MYSQL_PORT', 3306)),
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
        gemini_api_key = os.getenv("GEMINI_API")
        if not gemini_api_key:
            raise ValueError("GEMINI_API environment variable not found. Please check your .env file.")
        
        self.llm_client = genai.Client(api_key=gemini_api_key)
        self.model_config = {
            'primary': 'gemini-2.5-flash',
            'fallback': 'gemini-1.5-flash'
        }
        
        # Schema cache
        self.schema_cache = {}
        self.schema_cache_ttl = 3600  # 1 hour
        
        # Context management
        self.context_history = []
        self.max_context_length = 10
        
        # Initialize utility classes
        self.query_optimizer = QueryOptimizer()
        self.notification_manager = NotificationManager()
        self.schema_analyzer = SchemaAnalyzer()
        
        # Initialize agents
        self.context_agent = ContextAgent()
        self.sql_generation_agent = SQLGenerationAgent(self.llm_client, self.model_config)
        self.execution_agent = ExecutionAgent()
        
        # Query count for monitoring
        self.query_count = 0
        self.start_time = datetime.now()
    
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
            adapter = DatabaseFactory.create_connector(db_type, config)
            
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
        
        # Analyze schema for insights
        schema_insights = self.schema_analyzer.analyze_schema(schema)
        
        # Build relationship graph
        self._build_relationship_graph(schema)
        
        # Cache the schema
        self.schema_cache[cache_key] = (time.time(), schema)
        
        return schema
    
    def _build_relationship_graph(self, schema: DatabaseSchema):
        """Build a graph of table relationships"""
        if not hasattr(self, 'relationship_graph'):
            self.relationship_graph = nx.DiGraph()
        
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
        if hasattr(self, 'relationship_graph') and self.relationship_graph.number_of_nodes() > 0:
            for t1 in tables:
                for t2 in tables:
                    if t1 != t2 and self.relationship_graph.has_node(t1) and self.relationship_graph.has_node(t2):
                        try:
                            path = nx.shortest_path(self.relationship_graph, t1, t2)
                            if len(path) > 2:
                                relationships.append(f"Indirect: {' -> '.join(path)}")
                        except:
                            pass
        
        return "Join paths:\n" + "\n".join(f"  - {r}" for r in relationships) if relationships else "No direct relationships found"
    
    def _generate_fallback_sql(self, question: str, context: Dict, schema: DatabaseSchema) -> str:
        """Generate fallback SQL when main generation fails"""
        sql_parts = []
        
        # SELECT clause
        if context.get('requires_aggregation'):
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
            
            # Debug: print context structure
            print(f"🔍 Debug - Context keys: {list(context.keys())}")
            
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
            self._send_telegram_report(question, sql, result, time.time() - start_time, context)
            
            # Update context history
            self._update_context_history(question, sql, context, time.time() - start_time, True)
            
            print(f"✅ Completed in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)
            
            # Send error notification
            self._send_telegram_report(question, "N/A", None, time.time() - start_time, 
                                     {'error': str(e)}, "ERROR")
            
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
    
    def _send_telegram_report(self, question: str, sql: str, result: Dict, 
                             execution_time: float, context: Dict, status: str = "SUCCESS"):
        """Send comprehensive report to Telegram"""
        try:
            # Create structured notification
            notification = self.notification_manager.create_structured_notification(
                question=question,
                sql=sql,
                results=result,
                execution_time=execution_time,
                context=context,
                status=status
            )
            
            # Send to Telegram
            self.notification_manager.send_notification(
                notification, 
                channel='telegram', 
                notification_type=status
            )
            
        except Exception as e:
            print(f"Failed to send Telegram report: {e}")
    
    def _update_context_history(self, question: str, sql: str, context: Dict, 
                               execution_time: float, success: bool):
        """Update context history"""
        self.context_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'sql': sql,
            'context': context,
            'execution_time': execution_time,
            'success': success
        })
        
        # Trim history
        if len(self.context_history) > self.max_context_length:
            self.context_history = self.context_history[-self.max_context_length:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime': {
                'seconds': uptime_seconds,
                'formatted': f"{uptime_seconds / 3600:.1f} hours"
            },
            'queries': {
                'total': self.query_count,
                'quota_usage': f"{(self.query_count / 100) * 100:.1f}%",
                'remaining': max(0, 100 - self.query_count)
            },
            'context': {
                'history_items': len(self.context_history),
                'max_length': self.max_context_length
            },
            'model': {
                'current': self.model_config['primary'],
                'fallback': self.model_config['fallback']
            },
            'database': {
                'type': self.current_db_type,
                'connected': self.current_adapter is not None
            }
        }
