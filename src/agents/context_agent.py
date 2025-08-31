"""
Context analysis agent for understanding user queries
"""

import re
from typing import Dict, Any, List
from ..database.models import DatabaseSchema


class ContextAgent:
    """Analyze context and determine query complexity"""
    
    def __init__(self):
        self.language_patterns = {
            'vietnamese': 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ',
            'english': 'abcdefghijklmnopqrstuvwxyz'
        }
    
    def analyze_context(self, question: str, schema: DatabaseSchema) -> Dict[str, Any]:
        """Analyze context dynamically without hardcoded dependencies"""
        context_info = {
            'language': self._detect_language(question),
            'complexity': 'simple',
            'operation_type': 'read',
            'requires_join': False,
            'requires_aggregation': False,
            'time_based': False,
            'related_context': [],
            'detected_tables': [],
            'detected_columns': [],
            'query_pattern': 'unknown',
            'insertion_order': [],
            'constraint_aware': False
        }
        
        # Analyze question against actual schema
        if schema:
            context_info.update(self._analyze_question_against_schema(question, schema))
        
        # Find related context from history
        context_info['related_context'] = self._find_related_context(question)
        
        return context_info
    
    def _detect_language(self, question: str) -> str:
        """Detect language dynamically"""
        vietnamese_chars = self.language_patterns['vietnamese']
        return 'vi' if any(char in question for char in vietnamese_chars) else 'en'
    
    def _analyze_question_against_schema(self, question: str, schema: DatabaseSchema) -> Dict[str, Any]:
        """Analyze question against actual database schema"""
        question_lower = question.lower()
        analysis = {
            'detected_tables': [],
            'detected_columns': [],
            'query_pattern': 'unknown',
            'insertion_order': [],
            'constraint_aware': False
        }
        
        # Detect tables mentioned in question
        for table_name in schema.tables.keys():
            if table_name.lower() in question_lower:
                analysis['detected_tables'].append(table_name)
        
        # Detect columns mentioned
        for table_name, table_info in schema.tables.items():
            for column in table_info.columns:
                if column['name'].lower() in question_lower:
                    analysis['detected_columns'].append({
                        'table': table_name,
                        'column': column['name'],
                        'type': column['type']
                    })
        
        # Analyze complexity based on schema relationships
        if len(analysis['detected_tables']) > 1:
            analysis['requires_join'] = True
            analysis['complexity'] = 'complex'
        
        # Check for aggregation patterns
        agg_keywords = ['count', 'sum', 'average', 'max', 'min', 'total', 'đếm', 'tổng', 'trung bình']
        if any(keyword in question_lower for keyword in agg_keywords):
            analysis['requires_aggregation'] = True
            analysis['complexity'] = 'complex'
        
        # Check for write operations
        write_keywords = ['insert', 'update', 'delete', 'create', 'drop', 'alter', 'thêm', 'sửa', 'xóa', 'tạo']
        if any(keyword in question_lower for keyword in write_keywords):
            analysis['operation_type'] = 'write'
            analysis['constraint_aware'] = True
        
        # Check for time-based queries
        time_keywords = ['today', 'yesterday', 'last week', 'this month', 'hôm nay', 'tuần trước', 'tháng này']
        if any(keyword in question_lower for keyword in time_keywords):
            analysis['time_based'] = True
        
        # Determine query pattern
        analysis['query_pattern'] = self._determine_query_pattern(question_lower, analysis)
        
        # Determine insertion order for write operations
        if analysis['operation_type'] == 'write' and analysis['detected_tables']:
            analysis['insertion_order'] = self._determine_insertion_order(analysis['detected_tables'], schema)
        
        return analysis
    
    def _determine_query_pattern(self, question: str, analysis: Dict[str, Any]) -> str:
        """Determine the type of query pattern"""
        if analysis.get('requires_join'):
            return 'join_query'
        elif analysis.get('requires_aggregation'):
            return 'aggregation_query'
        elif analysis.get('time_based'):
            return 'time_based_query'
        elif analysis.get('operation_type') == 'write':
            return 'write_operation'
        elif 'top' in question or 'best' in question or 'most' in question:
            return 'ranking_query'
        else:
            return 'simple_select'
    
    def _determine_insertion_order(self, tables: List[str], schema: DatabaseSchema) -> List[str]:
        """Determine optimal insertion order based on foreign key constraints"""
        if len(tables) <= 1:
            return tables
        
        # Build dependency graph
        dependencies = {}
        for table in tables:
            dependencies[table] = []
            if table in schema.tables:
                for fk in schema.tables[table].foreign_keys:
                    if fk['referenced_table'] in tables:
                        dependencies[table].append(fk['referenced_table'])
        
        # Sort by dependency count (fewer dependencies first)
        sorted_tables = sorted(tables, key=lambda x: len(dependencies[x]))
        
        return sorted_tables
    
    def _find_related_context(self, question: str) -> List[Dict]:
        """Find contextually relevant history"""
        # This would be implemented with actual context history
        # For now, return empty list
        return []
