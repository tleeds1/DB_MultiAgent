"""
Schema analysis and relationship detection utilities
"""

import networkx as nx
from typing import Dict, List, Any, Tuple
from ..database.models import DatabaseSchema


class SchemaAnalyzer:
    """Analyze database schema and detect relationships"""
    
    def __init__(self):
        self.relationship_graph = nx.DiGraph()
    
    def analyze_schema(self, schema: DatabaseSchema) -> Dict[str, Any]:
        """Analyze schema and return insights"""
        # Build relationship graph
        self._build_relationship_graph(schema)
        
        # Detect implicit relationships
        implicit_relationships = self._detect_implicit_relationships(schema)
        
        # Analyze table dependencies
        dependencies = self._analyze_table_dependencies(schema)
        
        # Find circular references
        circular_refs = self._find_circular_references()
        
        return {
            'implicit_relationships': implicit_relationships,
            'dependencies': dependencies,
            'circular_references': circular_refs,
            'insertion_order': self._get_optimal_insertion_order(schema),
            'deletion_order': self._get_optimal_deletion_order(schema)
        }
    
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
    
    def _analyze_table_dependencies(self, schema: DatabaseSchema) -> Dict[str, List[str]]:
        """Analyze which tables depend on which other tables"""
        dependencies = {}
        
        for table_name in schema.tables:
            dependencies[table_name] = []
            
            # Check foreign key dependencies
            for fk in schema.tables[table_name].foreign_keys:
                if fk['referenced_table'] not in dependencies[table_name]:
                    dependencies[table_name].append(fk['referenced_table'])
            
            # Check implicit dependencies
            for rel in schema.relationships:
                if rel['from_table'] == table_name:
                    if rel['to_table'] not in dependencies[table_name]:
                        dependencies[table_name].append(rel['to_table'])
        
        return dependencies
    
    def _find_circular_references(self) -> List[List[str]]:
        """Find circular references in the schema"""
        try:
            cycles = list(nx.simple_cycles(self.relationship_graph))
            return cycles
        except nx.NetworkXNoPath:
            return []
    
    def _get_optimal_insertion_order(self, schema: DatabaseSchema) -> List[str]:
        """Get optimal order for inserting data to avoid constraint violations"""
        try:
            # Use topological sort to determine insertion order
            # Tables with no dependencies come first
            sorted_tables = list(nx.topological_sort(self.relationship_graph))
            return sorted_tables
        except nx.NetworkXError:
            # If there are cycles, fall back to dependency-based ordering
            return self._get_dependency_based_order(schema)
    
    def _get_optimal_deletion_order(self, schema: DatabaseSchema) -> List[str]:
        """Get optimal order for deleting data to avoid constraint violations"""
        try:
            # Reverse topological sort for deletion
            # Tables with dependencies come first
            sorted_tables = list(nx.topological_sort(self.relationship_graph))
            return list(reversed(sorted_tables))
        except nx.NetworkXError:
            # If there are cycles, fall back to reverse dependency-based ordering
            return list(reversed(self._get_dependency_based_order(schema)))
    
    def _get_dependency_based_order(self, schema: DatabaseSchema) -> List[str]:
        """Get table order based on dependency analysis"""
        dependencies = self._analyze_table_dependencies(schema)
        
        # Sort tables by number of dependencies (fewer dependencies first)
        sorted_tables = sorted(
            dependencies.keys(),
            key=lambda x: len(dependencies[x])
        )
        
        return sorted_tables
    
    def get_required_tables_for_insert(self, target_table: str, schema: DatabaseSchema) -> List[str]:
        """Get tables that must be populated before inserting into target table"""
        if target_table not in self.relationship_graph:
            return []
        
        # Find all tables that this table depends on
        required_tables = []
        
        # Check direct foreign key dependencies
        if target_table in schema.tables:
            for fk in schema.tables[target_table].foreign_keys:
                if fk['referenced_table'] not in required_tables:
                    required_tables.append(fk['referenced_table'])
        
        # Check relationship dependencies
        for rel in schema.relationships:
            if rel['from_table'] == target_table:
                if rel['to_table'] not in required_tables:
                    required_tables.append(rel['to_table'])
        
        # Recursively check dependencies of required tables
        all_required = required_tables.copy()
        for table in required_tables:
            sub_required = self.get_required_tables_for_insert(table, schema)
            for sub_table in sub_required:
                if sub_table not in all_required:
                    all_required.append(sub_table)
        
        return all_required
    
    def validate_insert_order(self, tables: List[str], schema: DatabaseSchema) -> Tuple[bool, List[str]]:
        """Validate if the proposed insert order is valid"""
        if not tables:
            return True, []
        
        errors = []
        inserted_tables = set()
        
        for table in tables:
            # Check if all required tables are already inserted
            required = self.get_required_tables_for_insert(table, schema)
            missing = [req for req in required if req not in inserted_tables]
            
            if missing:
                errors.append(f"Table '{table}' requires tables {missing} to be inserted first")
            
            inserted_tables.add(table)
        
        return len(errors) == 0, errors
