"""
Query optimization utilities
"""

from typing import Dict, Any
from ..database.models import DatabaseSchema


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
    
    def analyze_query_performance(self, sql: str, schema: DatabaseSchema) -> Dict[str, Any]:
        """Analyze query performance characteristics"""
        analysis = {
            'estimated_cost': 'unknown',
            'recommendations': [],
            'potential_issues': []
        }
        
        # Check for common performance issues
        if 'SELECT *' in sql.upper():
            analysis['recommendations'].append("Consider selecting only needed columns instead of SELECT *")
        
        if 'ORDER BY' in sql.upper() and 'LIMIT' not in sql.upper():
            analysis['recommendations'].append("Consider adding LIMIT clause when using ORDER BY")
        
        # Check table sizes for JOIN optimization
        if 'JOIN' in sql.upper():
            table_names = self._extract_table_names(sql)
            if len(table_names) > 1:
                table_sizes = [(name, schema.tables[name].row_count) for name in table_names if name in schema.tables]
                table_sizes.sort(key=lambda x: x[1])
                if table_sizes:
                    analysis['recommendations'].append(f"Consider joining from smallest table ({table_sizes[0][0]}) first")
        
        return analysis
    
    def _extract_table_names(self, sql: str) -> list:
        """Extract table names from SQL (simplified)"""
        # This is a simplified implementation
        # In production, use proper SQL parser
        tables = []
        sql_upper = sql.upper()
        
        # Look for FROM and JOIN clauses
        if 'FROM' in sql_upper:
            parts = sql_upper.split('FROM')
            for part in parts[1:]:
                if 'JOIN' in part:
                    join_parts = part.split('JOIN')
                    for join_part in join_parts:
                        table = join_part.strip().split()[0].strip()
                        if table and table not in tables:
                            tables.append(table)
                else:
                    table = part.strip().split()[0].strip()
                    if table and table not in tables:
                        tables.append(table)
        
        return tables
