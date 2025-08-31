"""
Execution agent for query execution and monitoring
"""

from typing import Dict, Any, Optional
from datetime import datetime


class ExecutionAgent:
    """Handle query execution and monitoring"""
    
    def __init__(self):
        self.execution_history = []
        self.max_history = 100
    
    def execute_query(self, sql: str, adapter) -> tuple:
        """Execute query using the provided adapter"""
        start_time = datetime.now()
        
        try:
            # Execute the query
            result, error = adapter.execute_query(sql)
            
            # Record execution
            execution_record = {
                'timestamp': start_time,
                'sql': sql,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'success': error is None,
                'error': error,
                'result_summary': self._create_result_summary(result)
            }
            
            self._add_to_history(execution_record)
            
            return result, error
            
        except Exception as e:
            execution_record = {
                'timestamp': start_time,
                'sql': sql,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'success': False,
                'error': str(e),
                'result_summary': 'Execution failed'
            }
            
            self._add_to_history(execution_record)
            return None, str(e)
    
    def _create_result_summary(self, result: Optional[Dict]) -> str:
        """Create a summary of the execution result"""
        if not result:
            return "No result"
        
        if 'rows' in result:
            return f"Retrieved {result.get('row_count', 0)} rows"
        elif 'affected_rows' in result:
            return f"Affected {result['affected_rows']} rows"
        else:
            return "Operation completed"
    
    def _add_to_history(self, record: Dict):
        """Add execution record to history"""
        self.execution_history.append(record)
        
        # Keep only recent history
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_execution_time': 0,
                'total_execution_time': 0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for record in self.execution_history if record['success'])
        failed = total - successful
        total_time = sum(record['execution_time'] for record in self.execution_history)
        avg_time = total_time / total if total > 0 else 0
        
        return {
            'total_executions': total,
            'successful_executions': successful,
            'failed_executions': failed,
            'average_execution_time': avg_time,
            'total_execution_time': total_time,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }
    
    def get_recent_executions(self, limit: int = 10) -> list:
        """Get recent execution records"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
