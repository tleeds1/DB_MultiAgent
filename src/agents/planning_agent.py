"""
Planning Agent with Function Calling for complex query decomposition
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class QueryStepType(Enum):
    """Types of query execution steps"""
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    SELECT = "select"


@dataclass
class QueryStep:
    """A single step in query execution plan"""
    step_type: QueryStepType
    description: str
    parameters: Dict[str, Any]
    estimated_cost: int = 1
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class QueryPlan:
    """Complete query execution plan"""
    steps: List[QueryStep]
    estimated_total_cost: int
    complexity_level: str  # simple, medium, complex
    requires_optimization: bool = False


class PlanningAgent:
    """Agent for planning complex query execution using function calling"""
    
    def __init__(self):
        self.available_functions = self._define_functions()
        self.plan_history = []
    
    def _define_functions(self) -> List[Dict[str, Any]]:
        """Define available functions for query planning"""
        return [
            {
                "name": "analyze_query_complexity",
                "description": "Analyze the complexity of a natural language query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query to analyze"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "identify_required_tables",
                "description": "Identify which tables are needed for the query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query"
                        },
                        "available_tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of available tables in the database"
                        }
                    },
                    "required": ["query", "available_tables"]
                }
            },
            {
                "name": "plan_filter_operations",
                "description": "Plan filtering operations for the query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query"
                        },
                        "target_tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tables to apply filters to"
                        },
                        "schema_info": {
                            "type": "object",
                            "description": "Schema information for the tables"
                        }
                    },
                    "required": ["query", "target_tables"]
                }
            },
            {
                "name": "plan_join_operations",
                "description": "Plan join operations between tables",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tables to join"
                        },
                        "relationships": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Known relationships between tables"
                        }
                    },
                    "required": ["tables"]
                }
            },
            {
                "name": "plan_aggregation_operations",
                "description": "Plan aggregation operations for the query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to aggregate"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "plan_sorting_operations",
                "description": "Plan sorting operations for the query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available columns for sorting"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "optimize_query_plan",
                "description": "Optimize the query execution plan for better performance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object",
                            "description": "The current query plan to optimize"
                        },
                        "schema_info": {
                            "type": "object",
                            "description": "Database schema information"
                        }
                    },
                    "required": ["plan"]
                }
            }
        ]
    
    def create_query_plan(self, question: str, context: Dict[str, Any], 
                         schema: Any) -> QueryPlan:
        """Create a comprehensive query execution plan"""
        
        # Step 1: Analyze query complexity
        complexity = self._analyze_complexity(question, context)
        
        # Step 2: Identify required tables
        required_tables = self._identify_tables(question, schema)
        
        # Step 3: Create execution steps
        steps = []
        
        # Add table selection step
        if required_tables:
            steps.append(QueryStep(
                step_type=QueryStepType.SELECT,
                description=f"Select data from tables: {', '.join(required_tables)}",
                parameters={"tables": required_tables},
                estimated_cost=1
            ))
        
        # Add filtering step if needed
        if context.get('requires_filtering', False):
            filter_step = self._create_filter_step(question, required_tables, schema)
            if filter_step:
                steps.append(filter_step)
        
        # Add join step if needed
        if context.get('requires_join', False) and len(required_tables) > 1:
            join_step = self._create_join_step(required_tables, schema)
            if join_step:
                steps.append(join_step)
        
        # Add aggregation step if needed
        if context.get('requires_aggregation', False):
            agg_step = self._create_aggregation_step(question, schema)
            if agg_step:
                steps.append(agg_step)
        
        # Add sorting step if needed
        if context.get('order_by_needed', False):
            sort_step = self._create_sorting_step(question, schema)
            if sort_step:
                steps.append(sort_step)
        
        # Add limit step if needed
        if context.get('limit_specified', False):
            limit_step = self._create_limit_step(context)
            if limit_step:
                steps.append(limit_step)
        
        # Calculate total cost
        total_cost = sum(step.estimated_cost for step in steps)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(total_cost, len(steps))
        
        # Create plan
        plan = QueryPlan(
            steps=steps,
            estimated_total_cost=total_cost,
            complexity_level=complexity_level,
            requires_optimization=total_cost > 5 or len(steps) > 3
        )
        
        # Store plan history
        self.plan_history.append({
            'timestamp': self._get_timestamp(),
            'question': question,
            'plan': asdict(plan),
            'context': context
        })
        
        return plan
    
    def _analyze_complexity(self, question: str, context: Dict[str, Any]) -> str:
        """Analyze query complexity"""
        complexity_score = 0
        
        # Count mentioned tables
        complexity_score += len(context.get('mentioned_tables', []))
        
        # Add points for different operations
        if context.get('requires_join'):
            complexity_score += 2
        if context.get('requires_aggregation'):
            complexity_score += 2
        if context.get('time_based'):
            complexity_score += 1
        if context.get('order_by_needed'):
            complexity_score += 1
        
        # Check for complex keywords
        complex_keywords = ['subquery', 'window', 'partition', 'recursive', 'cte']
        question_lower = question.lower()
        for keyword in complex_keywords:
            if keyword in question_lower:
                complexity_score += 3
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "medium"
        else:
            return "complex"
    
    def _identify_tables(self, question: str, schema: Any) -> List[str]:
        """Identify required tables for the query"""
        mentioned_tables = []
        
        if not schema or not hasattr(schema, 'tables'):
            return mentioned_tables
        
        available_tables = list(schema.tables.keys())
        question_lower = question.lower()
        
        # Direct table name matching
        for table in available_tables:
            table_lower = table.lower()
            if table_lower in question_lower:
                mentioned_tables.append(table)
            elif table_lower.rstrip('s') in question_lower:
                mentioned_tables.append(table)
        
        # Check for common table patterns
        common_patterns = {
            'user': ['users', 'user', 'customer', 'customers'],
            'order': ['orders', 'order', 'purchase', 'purchases'],
            'product': ['products', 'product', 'item', 'items'],
            'transaction': ['transactions', 'transaction', 'payment', 'payments']
        }
        
        for pattern, variations in common_patterns.items():
            if pattern in question_lower:
                for table in available_tables:
                    if any(var in table.lower() for var in variations):
                        if table not in mentioned_tables:
                            mentioned_tables.append(table)
        
        return mentioned_tables
    
    def _create_filter_step(self, question: str, tables: List[str], schema: Any) -> Optional[QueryStep]:
        """Create filtering step"""
        filters = []
        
        # Extract filter conditions from question
        question_lower = question.lower()
        
        # Date filters
        date_patterns = [
            r'today', r'yesterday', r'this week', r'last week',
            r'this month', r'last month', r'this year', r'last year',
            r'between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})',
            r'since (\d{4}-\d{2}-\d{2})',
            r'before (\d{4}-\d{2}-\d{2})',
            r'after (\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, question_lower)
            if match:
                filters.append({
                    'type': 'date',
                    'condition': match.group(0),
                    'tables': tables
                })
        
        # Numeric filters
        numeric_patterns = [
            r'greater than (\d+)',
            r'less than (\d+)',
            r'more than (\d+)',
            r'at least (\d+)',
            r'at most (\d+)'
        ]
        
        for pattern in numeric_patterns:
            match = re.search(pattern, question_lower)
            if match:
                filters.append({
                    'type': 'numeric',
                    'condition': match.group(0),
                    'value': match.group(1),
                    'tables': tables
                })
        
        if filters:
            return QueryStep(
                step_type=QueryStepType.FILTER,
                description=f"Apply filters: {', '.join(f['condition'] for f in filters)}",
                parameters={"filters": filters},
                estimated_cost=2
            )
        
        return None
    
    def _create_join_step(self, tables: List[str], schema: Any) -> Optional[QueryStep]:
        """Create join step"""
        if len(tables) < 2:
            return None
        
        # Find relationships between tables
        relationships = []
        if hasattr(schema, 'relationships'):
            for rel in schema.relationships:
                if rel['from_table'] in tables and rel['to_table'] in tables:
                    relationships.append(rel)
        
        if relationships:
            return QueryStep(
                step_type=QueryStepType.JOIN,
                description=f"Join tables: {' -> '.join(tables)}",
                parameters={
                    "tables": tables,
                    "relationships": relationships
                },
                estimated_cost=3
            )
        
        return None
    
    def _create_aggregation_step(self, question: str, schema: Any) -> Optional[QueryStep]:
        """Create aggregation step"""
        aggregations = []
        question_lower = question.lower()
        
        # Detect aggregation functions
        agg_functions = {
            'count': ['count', 'how many', 'number of'],
            'sum': ['sum', 'total', 'sum of'],
            'average': ['average', 'avg', 'mean'],
            'max': ['maximum', 'max', 'highest'],
            'min': ['minimum', 'min', 'lowest']
        }
        
        for func, keywords in agg_functions.items():
            if any(keyword in question_lower for keyword in keywords):
                aggregations.append(func)
        
        if aggregations:
            return QueryStep(
                step_type=QueryStepType.AGGREGATE,
                description=f"Apply aggregations: {', '.join(aggregations)}",
                parameters={"functions": aggregations},
                estimated_cost=2
            )
        
        return None
    
    def _create_sorting_step(self, question: str, schema: Any) -> Optional[QueryStep]:
        """Create sorting step"""
        question_lower = question.lower()
        
        # Detect sorting requirements
        sort_keywords = {
            'desc': ['highest', 'top', 'best', 'latest', 'newest', 'most'],
            'asc': ['lowest', 'bottom', 'worst', 'earliest', 'oldest', 'least']
        }
        
        sort_order = 'desc'  # default
        for order, keywords in sort_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                sort_order = order
                break
        
        return QueryStep(
            step_type=QueryStepType.SORT,
            description=f"Sort results in {sort_order}ending order",
            parameters={"order": sort_order},
            estimated_cost=1
        )
    
    def _create_limit_step(self, context: Dict[str, Any]) -> Optional[QueryStep]:
        """Create limit step"""
        limit_value = context.get('limit_value', 100)
        
        return QueryStep(
            step_type=QueryStepType.LIMIT,
            description=f"Limit results to {limit_value} rows",
            parameters={"limit": limit_value},
            estimated_cost=1
        )
    
    def _determine_complexity_level(self, total_cost: int, step_count: int) -> str:
        """Determine complexity level based on cost and steps"""
        if total_cost <= 3 and step_count <= 2:
            return "simple"
        elif total_cost <= 7 and step_count <= 4:
            return "medium"
        else:
            return "complex"
    
    def optimize_plan(self, plan: QueryPlan, schema: Any) -> QueryPlan:
        """Optimize the query plan for better performance"""
        optimized_steps = []
        
        # Reorder steps for better performance
        # 1. Filter first (reduce data set)
        filter_steps = [s for s in plan.steps if s.step_type == QueryStepType.FILTER]
        optimized_steps.extend(filter_steps)
        
        # 2. Select tables
        select_steps = [s for s in plan.steps if s.step_type == QueryStepType.SELECT]
        optimized_steps.extend(select_steps)
        
        # 3. Join tables
        join_steps = [s for s in plan.steps if s.step_type == QueryStepType.JOIN]
        optimized_steps.extend(join_steps)
        
        # 4. Aggregate
        agg_steps = [s for s in plan.steps if s.step_type == QueryStepType.AGGREGATE]
        optimized_steps.extend(agg_steps)
        
        # 5. Sort
        sort_steps = [s for s in plan.steps if s.step_type == QueryStepType.SORT]
        optimized_steps.extend(sort_steps)
        
        # 6. Limit last
        limit_steps = [s for s in plan.steps if s.step_type == QueryStepType.LIMIT]
        optimized_steps.extend(limit_steps)
        
        # Recalculate total cost
        total_cost = sum(step.estimated_cost for step in optimized_steps)
        
        return QueryPlan(
            steps=optimized_steps,
            estimated_total_cost=total_cost,
            complexity_level=plan.complexity_level,
            requires_optimization=False
        )
    
    def generate_sql_from_plan(self, plan: QueryPlan) -> str:
        """Generate SQL from the execution plan"""
        sql_parts = []
        
        # Build SELECT clause
        select_clause = "SELECT *"
        sql_parts.append(select_clause)
        
        # Build FROM clause
        from_tables = []
        for step in plan.steps:
            if step.step_type == QueryStepType.SELECT:
                from_tables.extend(step.parameters.get('tables', []))
        
        if from_tables:
            sql_parts.append(f"FROM {', '.join(from_tables)}")
        
        # Build WHERE clause
        where_conditions = []
        for step in plan.steps:
            if step.step_type == QueryStepType.FILTER:
                filters = step.parameters.get('filters', [])
                for filter_info in filters:
                    if filter_info['type'] == 'date':
                        where_conditions.append(f"created_at >= CURRENT_DATE - INTERVAL '1 day'")
                    elif filter_info['type'] == 'numeric':
                        value = filter_info['value']
                        where_conditions.append(f"amount > {value}")
        
        if where_conditions:
            sql_parts.append(f"WHERE {' AND '.join(where_conditions)}")
        
        # Build GROUP BY clause
        for step in plan.steps:
            if step.step_type == QueryStepType.AGGREGATE:
                sql_parts.append("GROUP BY id")
        
        # Build ORDER BY clause
        for step in plan.steps:
            if step.step_type == QueryStepType.SORT:
                order = step.parameters.get('order', 'desc')
                sql_parts.append(f"ORDER BY id {order.upper()}")
        
        # Build LIMIT clause
        for step in plan.steps:
            if step.step_type == QueryStepType.LIMIT:
                limit = step.parameters.get('limit', 100)
                sql_parts.append(f"LIMIT {limit}")
        
        return " ".join(sql_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_plan_summary(self, plan: QueryPlan) -> Dict[str, Any]:
        """Get summary of the query plan"""
        return {
            'total_steps': len(plan.steps),
            'estimated_cost': plan.estimated_total_cost,
            'complexity': plan.complexity_level,
            'step_types': [step.step_type.value for step in plan.steps],
            'optimization_needed': plan.requires_optimization
        }
