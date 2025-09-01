"""
Verification Agent for validating query results against business logic
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class BusinessRule:
    """Business rule for validation"""
    name: str
    condition: str
    description: str
    severity: str = "warning"  # warning, error, critical


@dataclass
class ValidationResult:
    """Result of validation check"""
    rule_name: str
    passed: bool
    message: str
    severity: str
    suggested_fix: Optional[str] = None


class VerificationAgent:
    """Agent for verifying query results against business logic"""
    
    def __init__(self):
        self.business_rules = self._initialize_business_rules()
        self.validation_history = []
    
    def _initialize_business_rules(self) -> Dict[str, List[BusinessRule]]:
        """Initialize common business rules"""
        return {
            'general': [
                BusinessRule(
                    name="non_empty_result",
                    condition="row_count > 0",
                    description="Query should return at least one result",
                    severity="warning"
                ),
                BusinessRule(
                    name="reasonable_limit",
                    condition="row_count <= 10000",
                    description="Query should not return more than 10,000 rows",
                    severity="warning"
                )
            ],
            'financial': [
                BusinessRule(
                    name="positive_amounts",
                    condition="all_amounts >= 0",
                    description="Financial amounts should be non-negative",
                    severity="error"
                ),
                BusinessRule(
                    name="reasonable_amounts",
                    condition="max_amount <= 1000000",
                    description="Amounts should be within reasonable range",
                    severity="warning"
                )
            ],
            'temporal': [
                BusinessRule(
                    name="future_dates_valid",
                    condition="no_future_dates",
                    description="Dates should not be in the future",
                    severity="error"
                ),
                BusinessRule(
                    name="reasonable_date_range",
                    condition="date_range <= 365",
                    description="Date range should be reasonable",
                    severity="warning"
                )
            ]
        }
    
    def verify_query_result(self, result: Dict[str, Any], question: str, 
                          context: Dict[str, Any], schema: Any) -> List[ValidationResult]:
        """Verify query result against business rules"""
        validations = []
        
        # Basic result validation
        validations.extend(self._validate_basic_result(result))
        
        # Context-based validation
        validations.extend(self._validate_context(result, context))
        
        # Schema-based validation
        validations.extend(self._validate_schema(result, schema))
        
        # Question-specific validation
        validations.extend(self._validate_question_specific(result, question))
        
        # Store validation history
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'validations': [v.__dict__ for v in validations],
            'overall_passed': all(v.passed for v in validations)
        })
        
        return validations
    
    def _validate_basic_result(self, result: Dict[str, Any]) -> List[ValidationResult]:
        """Validate basic result properties"""
        validations = []
        
        # Check if result exists
        if not result:
            validations.append(ValidationResult(
                rule_name="result_exists",
                passed=False,
                message="Query returned no result",
                severity="error"
            ))
            return validations
        
        # Check row count
        row_count = result.get('row_count', 0)
        if row_count == 0:
            validations.append(ValidationResult(
                rule_name="non_empty_result",
                passed=False,
                message="Query returned no rows",
                severity="warning",
                suggested_fix="Consider checking if the WHERE conditions are too restrictive"
            ))
        elif row_count > 10000:
            validations.append(ValidationResult(
                rule_name="reasonable_limit",
                passed=False,
                message=f"Query returned {row_count} rows (very large result set)",
                severity="warning",
                suggested_fix="Consider adding LIMIT clause or more specific filters"
            ))
        
        return validations
    
    def _validate_context(self, result: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationResult]:
        """Validate based on query context"""
        validations = []
        
        # Check if aggregation was expected but not performed
        if context.get('requires_aggregation') and 'rows' in result:
            # Check if result has aggregation-like structure
            has_aggregation = any(
                col.lower() in ['count', 'sum', 'avg', 'max', 'min', 'total']
                for col in result.get('columns', [])
            )
            
            if not has_aggregation:
                validations.append(ValidationResult(
                    rule_name="aggregation_expected",
                    passed=False,
                    message="Aggregation was expected but not found in result",
                    severity="warning",
                    suggested_fix="Consider using GROUP BY with aggregation functions"
                ))
        
        # Check if join was expected but not performed
        if context.get('requires_join') and 'rows' in result:
            # Simple heuristic: if multiple tables mentioned but result seems single-table
            mentioned_tables = context.get('mentioned_tables', [])
            if len(mentioned_tables) > 1:
                # Check if result columns suggest single table
                columns = result.get('columns', [])
                table_prefixes = set()
                for col in columns:
                    if '.' in col:
                        table_prefixes.add(col.split('.')[0])
                
                if len(table_prefixes) <= 1:
                    validations.append(ValidationResult(
                        rule_name="join_expected",
                        passed=False,
                        message="Join was expected but result appears to be from single table",
                        severity="warning",
                        suggested_fix="Consider using JOIN to combine data from multiple tables"
                    ))
        
        return validations
    
    def _validate_schema(self, result: Dict[str, Any], schema: Any) -> List[ValidationResult]:
        """Validate against database schema"""
        validations = []
        
        if not schema or 'rows' not in result:
            return validations
        
        # Check if returned columns exist in schema
        result_columns = result.get('columns', [])
        schema_tables = getattr(schema, 'tables', {})
        
        for col in result_columns:
            col_name = col.split('.')[-1] if '.' in col else col
            found_in_schema = False
            
            for table_info in schema_tables.values():
                table_columns = [c['name'] for c in table_info.columns]
                if col_name in table_columns:
                    found_in_schema = True
                    break
            
            if not found_in_schema:
                validations.append(ValidationResult(
                    rule_name="column_exists",
                    passed=False,
                    message=f"Column '{col}' not found in schema",
                    severity="error",
                    suggested_fix="Check column name spelling and table references"
                ))
        
        return validations
    
    def _validate_question_specific(self, result: Dict[str, Any], question: str) -> List[ValidationResult]:
        """Validate based on specific question patterns"""
        validations = []
        question_lower = question.lower()
        
        # Financial queries validation
        if any(word in question_lower for word in ['revenue', 'sales', 'amount', 'price', 'cost']):
            validations.extend(self._validate_financial_data(result))
        
        # Date/time queries validation
        if any(word in question_lower for word in ['date', 'time', 'created', 'updated', 'when']):
            validations.extend(self._validate_temporal_data(result))
        
        # Count queries validation
        if 'count' in question_lower or 'how many' in question_lower:
            validations.extend(self._validate_count_data(result))
        
        return validations
    
    def _validate_financial_data(self, result: Dict[str, Any]) -> List[ValidationResult]:
        """Validate financial data"""
        validations = []
        
        if 'rows' not in result:
            return validations
        
        rows = result['rows']
        columns = result.get('columns', [])
        
        # Find amount/price columns
        amount_columns = []
        for i, col in enumerate(columns):
            col_lower = col.lower()
            if any(word in col_lower for word in ['amount', 'price', 'cost', 'revenue', 'sales']):
                amount_columns.append(i)
        
        if amount_columns:
            for row in rows[:100]:  # Check first 100 rows
                for col_idx in amount_columns:
                    if col_idx < len(row):
                        value = row[col_idx]
                        if value is not None:
                            try:
                                num_value = float(value)
                                if num_value < 0:
                                    validations.append(ValidationResult(
                                        rule_name="positive_amounts",
                                        passed=False,
                                        message=f"Negative amount found: {num_value}",
                                        severity="error"
                                    ))
                                elif num_value > 1000000:
                                    validations.append(ValidationResult(
                                        rule_name="reasonable_amounts",
                                        passed=False,
                                        message=f"Unusually large amount: {num_value}",
                                        severity="warning"
                                    ))
                            except (ValueError, TypeError):
                                pass
        
        return validations
    
    def _validate_temporal_data(self, result: Dict[str, Any]) -> List[ValidationResult]:
        """Validate temporal data"""
        validations = []
        
        if 'rows' not in result:
            return validations
        
        rows = result['rows']
        columns = result.get('columns', [])
        
        # Find date columns
        date_columns = []
        for i, col in enumerate(columns):
            col_lower = col.lower()
            if any(word in col_lower for word in ['date', 'created', 'updated', 'time']):
                date_columns.append(i)
        
        if date_columns:
            current_time = datetime.now()
            for row in rows[:100]:  # Check first 100 rows
                for col_idx in date_columns:
                    if col_idx < len(row):
                        value = row[col_idx]
                        if value is not None:
                            try:
                                if isinstance(value, str):
                                    # Try to parse date string
                                    date_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                elif isinstance(value, datetime):
                                    date_value = value
                                else:
                                    continue
                                
                                if date_value > current_time:
                                    validations.append(ValidationResult(
                                        rule_name="future_dates_valid",
                                        passed=False,
                                        message=f"Future date found: {date_value}",
                                        severity="error"
                                    ))
                            except (ValueError, TypeError):
                                pass
        
        return validations
    
    def _validate_count_data(self, result: Dict[str, Any]) -> List[ValidationResult]:
        """Validate count data"""
        validations = []
        
        if 'rows' not in result:
            return validations
        
        rows = result['rows']
        columns = result.get('columns', [])
        
        # Check if result looks like a count
        if len(rows) == 1 and len(columns) == 1:
            try:
                count_value = int(rows[0][0])
                if count_value < 0:
                    validations.append(ValidationResult(
                        rule_name="valid_count",
                        passed=False,
                        message=f"Invalid count value: {count_value}",
                        severity="error"
                    ))
            except (ValueError, TypeError):
                pass
        
        return validations
    
    def get_validation_summary(self, validations: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        total = len(validations)
        passed = sum(1 for v in validations if v.passed)
        failed = total - passed
        
        errors = [v for v in validations if not v.passed and v.severity == 'error']
        warnings = [v for v in validations if not v.passed and v.severity == 'warning']
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'errors': len(errors),
            'warnings': len(warnings),
            'overall_passed': failed == 0,
            'critical_issues': len([v for v in validations if not v.passed and v.severity == 'critical'])
        }
    
    def suggest_improvements(self, validations: List[ValidationResult]) -> List[str]:
        """Suggest improvements based on validation failures"""
        suggestions = []
        
        for validation in validations:
            if not validation.passed and validation.suggested_fix:
                suggestions.append(f"{validation.rule_name}: {validation.suggested_fix}")
        
        return suggestions
