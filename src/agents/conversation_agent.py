"""
Conversation Agent for maintaining context and providing suggestions
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum


class ConversationType(Enum):
    """Types of conversation interactions"""
    QUERY = "query"
    CLARIFICATION = "clarification"
    SUGGESTION = "suggestion"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    timestamp: str
    user_input: str
    conversation_type: ConversationType
    context: Dict[str, Any]
    response: str
    sql_generated: Optional[str] = None
    execution_time: Optional[float] = None
    success: bool = True
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ConversationContext:
    """Maintained conversation context"""
    session_id: str
    start_time: str
    turns: List[ConversationTurn]
    current_topic: str
    mentioned_tables: List[str]
    mentioned_columns: List[str]
    query_patterns: List[str]
    user_preferences: Dict[str, Any]


class ConversationAgent:
    """Agent for managing conversation context and providing suggestions"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, ConversationContext] = {}
        self.suggestion_templates = self._initialize_suggestion_templates()
        self.context_patterns = self._initialize_context_patterns()
    
    def _initialize_suggestion_templates(self) -> Dict[str, List[str]]:
        """Initialize suggestion templates"""
        return {
            'follow_up': [
                "Dựa trên kết quả này, bạn có muốn xem thêm chi tiết về {table} không?",
                "Bạn có muốn so sánh với dữ liệu của tháng trước không?",
                "Có muốn tôi tạo biểu đồ cho kết quả này không?",
                "Bạn có muốn lọc thêm theo {column} không?"
            ],
            'clarification': [
                "Bạn có thể làm rõ hơn về {term} không?",
                "Bạn muốn xem dữ liệu của bảng nào cụ thể?",
                "Thời gian bạn muốn xem là khoảng nào?",
                "Bạn muốn kết quả được sắp xếp theo tiêu chí nào?"
            ],
            'optimization': [
                "Truy vấn này có thể được tối ưu hóa. Bạn có muốn tôi thử cách khác không?",
                "Kết quả khá lớn, bạn có muốn giới hạn số lượng không?",
                "Có vẻ như cần JOIN thêm bảng. Bạn có muốn tôi thử không?"
            ],
            'error_recovery': [
                "Có vẻ như có lỗi. Bạn có thể thử cách diễn đạt khác không?",
                "Tôi không tìm thấy bảng {table}. Bạn có thể kiểm tra tên bảng không?",
                "Có vẻ như cú pháp không đúng. Bạn có thể nói rõ hơn không?"
            ]
        }
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context patterns for understanding conversation flow"""
        return {
            'continuation': [
                r'và\s+(.+?)',
                r'thêm\s+(.+?)',
                r'cũng\s+(.+?)',
                r'nữa\s+(.+?)',
                r'còn\s+(.+?)'
            ],
            'clarification': [
                r'làm\s+rõ\s+(.+?)',
                r'giải\s+thích\s+(.+?)',
                r'có\s+nghĩa\s+là\s+(.+?)',
                r'ý\s+tôi\s+là\s+(.+?)'
            ],
            'comparison': [
                r'so\s+sánh\s+(.+?)',
                r'khác\s+biệt\s+(.+?)',
                r'tương\s+tự\s+(.+?)',
                r'giống\s+như\s+(.+?)'
            ],
            'time_based': [
                r'trong\s+(\d+)\s+(ngày|tuần|tháng|năm)',
                r'từ\s+(.+?)\s+đến\s+(.+?)',
                r'trước\s+(.+?)',
                r'sau\s+(.+?)'
            ]
        }
    
    def start_conversation(self, session_id: str, initial_context: Dict[str, Any] = None) -> ConversationContext:
        """Start a new conversation session"""
        if initial_context is None:
            initial_context = {}
        
        conversation = ConversationContext(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            turns=[],
            current_topic="general",
            mentioned_tables=initial_context.get('mentioned_tables', []),
            mentioned_columns=initial_context.get('mentioned_columns', []),
            query_patterns=[],
            user_preferences=initial_context.get('user_preferences', {})
        )
        
        self.conversations[session_id] = conversation
        return conversation
    
    def add_turn(self, session_id: str, user_input: str, response: str, 
                context: Dict[str, Any], conversation_type: ConversationType = ConversationType.QUERY,
                sql_generated: str = None, execution_time: float = None, 
                success: bool = True) -> ConversationTurn:
        """Add a new turn to the conversation"""
        
        if session_id not in self.conversations:
            self.start_conversation(session_id)
        
        conversation = self.conversations[session_id]
        
        # Generate suggestions based on context
        suggestions = self._generate_suggestions(user_input, context, conversation)
        
        # Create turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            conversation_type=conversation_type,
            context=context,
            response=response,
            sql_generated=sql_generated,
            execution_time=execution_time,
            success=success,
            suggestions=suggestions
        )
        
        # Add to conversation
        conversation.turns.append(turn)
        
        # Update conversation context
        self._update_conversation_context(conversation, turn, context)
        
        # Trim history if needed
        if len(conversation.turns) > self.max_history:
            conversation.turns = conversation.turns[-self.max_history:]
        
        return turn
    
    def _generate_suggestions(self, user_input: str, context: Dict[str, Any], 
                            conversation: ConversationContext) -> List[str]:
        """Generate contextual suggestions"""
        suggestions = []
        
        # Analyze input for patterns
        input_lower = user_input.lower()
        
        # Check for follow-up opportunities
        if context.get('success', True) and context.get('row_count', 0) > 0:
            mentioned_tables = context.get('mentioned_tables', [])
            if mentioned_tables:
                table = mentioned_tables[0]
                suggestions.append(
                    f"Dựa trên kết quả này, bạn có muốn xem thêm chi tiết về {table} không?"
                )
        
        # Check for optimization opportunities
        if context.get('execution_time', 0) > 2.0:  # Slow query
            suggestions.append(
                "Truy vấn này hơi chậm. Bạn có muốn tôi tối ưu hóa không?"
            )
        
        # Check for large result sets
        if context.get('row_count', 0) > 1000:
            suggestions.append(
                "Kết quả khá lớn. Bạn có muốn giới hạn số lượng hoặc tạo báo cáo không?"
            )
        
        # Check for aggregation opportunities
        if not context.get('requires_aggregation') and any(word in input_lower for word in ['tổng', 'trung bình', 'nhiều nhất', 'ít nhất']):
            suggestions.append(
                "Bạn có muốn tôi tính toán tổng hợp dữ liệu không?"
            )
        
        # Check for visualization opportunities
        if context.get('row_count', 0) > 10 and context.get('row_count', 0) < 100:
            suggestions.append(
                "Bạn có muốn tôi tạo biểu đồ cho kết quả này không?"
            )
        
        # Check for time-based analysis
        if any(word in input_lower for word in ['hôm nay', 'tuần này', 'tháng này', 'năm nay']):
            suggestions.append(
                "Bạn có muốn so sánh với cùng kỳ năm trước không?"
            )
        
        # Check for error recovery
        if not context.get('success', True):
            suggestions.extend([
                "Có vẻ như có lỗi. Bạn có thể thử cách diễn đạt khác không?",
                "Bạn có thể kiểm tra lại tên bảng hoặc cột không?"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _update_conversation_context(self, conversation: ConversationContext, 
                                   turn: ConversationTurn, context: Dict[str, Any]):
        """Update conversation context based on new turn"""
        
        # Update mentioned tables
        new_tables = context.get('mentioned_tables', [])
        for table in new_tables:
            if table not in conversation.mentioned_tables:
                conversation.mentioned_tables.append(table)
        
        # Update mentioned columns
        new_columns = context.get('mentioned_columns', [])
        for col in new_columns:
            col_name = col.get('column', col) if isinstance(col, dict) else col
            if col_name not in conversation.mentioned_columns:
                conversation.mentioned_columns.append(col_name)
        
        # Update query patterns
        pattern = self._extract_query_pattern(turn.user_input)
        if pattern and pattern not in conversation.query_patterns:
            conversation.query_patterns.append(pattern)
        
        # Update current topic
        conversation.current_topic = self._identify_topic(turn.user_input, context)
    
    def _extract_query_pattern(self, user_input: str) -> Optional[str]:
        """Extract query pattern from user input"""
        input_lower = user_input.lower()
        
        # Simple pattern extraction
        if any(word in input_lower for word in ['bao nhiêu', 'count', 'how many']):
            return "count_query"
        elif any(word in input_lower for word in ['tổng', 'sum', 'total']):
            return "sum_query"
        elif any(word in input_lower for word in ['trung bình', 'average', 'avg']):
            return "average_query"
        elif any(word in input_lower for word in ['danh sách', 'list', 'show']):
            return "list_query"
        elif any(word in input_lower for word in ['so sánh', 'compare']):
            return "comparison_query"
        elif any(word in input_lower for word in ['biểu đồ', 'chart', 'graph']):
            return "visualization_query"
        
        return None
    
    def _identify_topic(self, user_input: str, context: Dict[str, Any]) -> str:
        """Identify the current conversation topic"""
        input_lower = user_input.lower()
        
        # Topic identification based on keywords
        if any(word in input_lower for word in ['doanh thu', 'revenue', 'sales']):
            return "revenue_analysis"
        elif any(word in input_lower for word in ['khách hàng', 'customer', 'user']):
            return "customer_analysis"
        elif any(word in input_lower for word in ['sản phẩm', 'product', 'item']):
            return "product_analysis"
        elif any(word in input_lower for word in ['đơn hàng', 'order', 'transaction']):
            return "order_analysis"
        elif any(word in input_lower for word in ['thời gian', 'time', 'date']):
            return "time_analysis"
        
        return "general"
    
    def get_conversation_history(self, session_id: str, limit: int = 5) -> List[ConversationTurn]:
        """Get recent conversation history"""
        if session_id not in self.conversations:
            return []
        
        conversation = self.conversations[session_id]
        return conversation.turns[-limit:] if limit > 0 else conversation.turns
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation context"""
        if session_id not in self.conversations:
            return {}
        
        conversation = self.conversations[session_id]
        
        return {
            'session_id': conversation.session_id,
            'start_time': conversation.start_time,
            'total_turns': len(conversation.turns),
            'current_topic': conversation.current_topic,
            'mentioned_tables': conversation.mentioned_tables,
            'mentioned_columns': conversation.mentioned_columns,
            'query_patterns': conversation.query_patterns,
            'user_preferences': conversation.user_preferences,
            'recent_queries': [
                {
                    'input': turn.user_input,
                    'type': turn.conversation_type.value,
                    'success': turn.success,
                    'timestamp': turn.timestamp
                }
                for turn in conversation.turns[-3:]  # Last 3 turns
            ]
        }
    
    def generate_context_prompt(self, session_id: str, current_question: str) -> str:
        """Generate context-aware prompt for LLM"""
        if session_id not in self.conversations:
            return current_question
        
        conversation = self.conversations[session_id]
        context_summary = self.get_context_summary(session_id)
        
        # Build context prompt
        context_parts = []
        
        # Add recent conversation history
        recent_turns = conversation.turns[-2:]  # Last 2 turns
        if recent_turns:
            context_parts.append("Recent conversation:")
            for turn in recent_turns:
                context_parts.append(f"User: {turn.user_input}")
                if turn.sql_generated:
                    context_parts.append(f"SQL: {turn.sql_generated}")
                context_parts.append(f"Result: {turn.response[:100]}...")
        
        # Add current topic
        if conversation.current_topic != "general":
            context_parts.append(f"Current topic: {conversation.current_topic}")
        
        # Add mentioned tables
        if conversation.mentioned_tables:
            context_parts.append(f"Previously mentioned tables: {', '.join(conversation.mentioned_tables)}")
        
        # Add query patterns
        if conversation.query_patterns:
            context_parts.append(f"Query patterns: {', '.join(conversation.query_patterns)}")
        
        # Combine context with current question
        if context_parts:
            context_text = "\n".join(context_parts)
            return f"Context:\n{context_text}\n\nCurrent question: {current_question}"
        
        return current_question
    
    def get_smart_suggestions(self, session_id: str, current_context: Dict[str, Any]) -> List[str]:
        """Get smart suggestions based on conversation history and current context"""
        if session_id not in self.conversations:
            return []
        
        conversation = self.conversations[session_id]
        suggestions = []
        
        # Get recent successful queries
        recent_successful = [
            turn for turn in conversation.turns[-3:]
            if turn.success and turn.conversation_type == ConversationType.QUERY
        ]
        
        if recent_successful:
            last_query = recent_successful[-1]
            
            # Suggest related queries
            if 'count' in last_query.user_input.lower():
                suggestions.append("Bạn có muốn xem chi tiết thay vì chỉ đếm số lượng không?")
            
            if 'list' in last_query.user_input.lower() or 'show' in last_query.user_input.lower():
                suggestions.append("Bạn có muốn tôi tạo báo cáo tổng hợp không?")
            
            if last_query.context.get('row_count', 0) > 10:
                suggestions.append("Bạn có muốn lọc kết quả theo tiêu chí cụ thể không?")
        
        # Suggest based on current topic
        if conversation.current_topic == "revenue_analysis":
            suggestions.append("Bạn có muốn phân tích xu hướng doanh thu theo thời gian không?")
        elif conversation.current_topic == "customer_analysis":
            suggestions.append("Bạn có muốn phân tích hành vi khách hàng không?")
        elif conversation.current_topic == "product_analysis":
            suggestions.append("Bạn có muốn phân tích hiệu suất sản phẩm không?")
        
        # Suggest based on mentioned tables
        if len(conversation.mentioned_tables) > 1:
            suggestions.append("Bạn có muốn tôi tạo báo cáo tổng hợp từ nhiều bảng không?")
        
        return suggestions[:3]
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_conversation_stats(self, session_id: str) -> Dict[str, Any]:
        """Get conversation statistics"""
        if session_id not in self.conversations:
            return {}
        
        conversation = self.conversations[session_id]
        
        # Calculate stats
        total_turns = len(conversation.turns)
        successful_queries = sum(1 for turn in conversation.turns if turn.success)
        avg_execution_time = sum(turn.execution_time or 0 for turn in conversation.turns) / total_turns if total_turns > 0 else 0
        
        return {
            'total_turns': total_turns,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_turns if total_turns > 0 else 0,
            'average_execution_time': avg_execution_time,
            'unique_tables_mentioned': len(conversation.mentioned_tables),
            'unique_columns_mentioned': len(conversation.mentioned_columns),
            'query_patterns_used': len(conversation.query_patterns)
        }
