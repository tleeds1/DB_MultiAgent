"""
SQL generation agent using LLM
"""

import re
import json
from typing import Dict, Any, List
from ..database.models import DatabaseSchema


class SQLGenerationAgent:
    """Generate SQL using schema understanding and context"""
    
    def __init__(self, llm_client, model_config: Dict[str, str]):
        self.llm_client = llm_client
        self.model_config = model_config
    
    def generate_sql(self, question: str, context: Dict[str, Any], 
                     schema: DatabaseSchema, db_type: str) -> str:
        """Generate SQL using schema understanding and context"""
        
        # Create detailed schema description
        schema_description = self._create_schema_description(schema, context)
        
        # Build relationship context
        relationship_context = self._build_relationship_context(context['detected_tables'], schema)
        
        # Create constraint-aware instructions
        constraint_instructions = self._create_constraint_instructions(context, schema)
        
        prompt = f"""
        You are an expert SQL developer. Generate an optimal SQL query based on the following:
        
        Database Type: {db_type}
        
        User Question: {question}
        
        Context Analysis:
        {json.dumps(context, indent=2)}
        
        Relevant Schema:
        {schema_description}
        
        Table Relationships:
        {relationship_context}
        
        Constraint Handling:
        {constraint_instructions}
        
        Rules:
        1. Use appropriate SQL dialect for {db_type}
        2. Only use tables and columns that exist in the schema
        3. Handle NULL values appropriately
        4. Use proper JOIN types based on relationships
        5. Apply appropriate WHERE clauses for filtering
        6. Use window functions for ranking queries if needed
        7. Add LIMIT clause if not specified but seems needed
        8. Optimize for performance using available indexes
        9. For write operations that may affect related tables, generate multiple SQL statements if necessary, separated by ';'. Ensure the order respects foreign key dependencies (parent tables first for inserts).
        10. If the query involves inserting data into multiple related tables, use Common Table Expressions (CTE) to insert into parent tables first and retrieve IDs for child tables to avoid constraint violations and ensure atomicity.
        
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
            
            return sql
            
        except Exception as e:
            print(f"SQL generation error: {e}")
            # Fallback to simple query generation
            return self._generate_fallback_sql(question, context, schema)
    
    def _create_schema_description(self, schema: DatabaseSchema, context: Dict) -> str:
        """Create focused schema description based on context"""
        description = []
        
        # If specific tables mentioned, focus on those
        tables_to_describe = context['detected_tables'] if context['detected_tables'] else list(schema.tables.keys())[:5]
        
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
        
        return "No direct relationships found"
    
    def _create_constraint_instructions(self, context: Dict, schema: DatabaseSchema) -> str:
        """Create instructions for handling constraints"""
        if context.get('operation_type') != 'write':
            return "No constraint handling needed for read operations"
        
        if not context.get('detected_tables'):
            return "No specific tables mentioned for constraint analysis"
        
        instructions = []
        
        # Analyze dependencies for mentioned tables
        for table_name in context['detected_tables']:
            if table_name in schema.tables:
                table_info = schema.tables[table_name]
                if table_info.foreign_keys:
                    instructions.append(f"Table '{table_name}' depends on:")
                    for fk in table_info.foreign_keys:
                        instructions.append(f"  - {fk['referenced_table']} (via {fk['column']})")
        
        if context.get('insertion_order'):
            instructions.append(f"\nRecommended insertion order: {' -> '.join(context['insertion_order'])}")
        
        return "\n".join(instructions) if instructions else "No constraint dependencies detected"
    
    def _generate_fallback_sql(self, question: str, context: Dict, schema: DatabaseSchema) -> str:
        """Generate fallback SQL when main generation fails"""
        sql_parts = []
        
        # SELECT clause
        if context.get('requires_aggregation'):
            sql_parts.append("SELECT COUNT(*) as count")
        else:
            sql_parts.append("SELECT *")
        
        # FROM clause
        if context['detected_tables']:
            sql_parts.append(f"FROM {context['detected_tables'][0]}")
        else:
            # Use first table in schema
            sql_parts.append(f"FROM {list(schema.tables.keys())[0]}")
        
        # Add LIMIT
        if context.get('limit_value'):
            sql_parts.append(f"LIMIT {context['limit_value']}")
        else:
            sql_parts.append("LIMIT 100")
        
        return " ".join(sql_parts)
