import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Optional, Any

load_dotenv()

app = FastAPI(title="MCP Server", version="0.1.0")

# Shared agent system
_agent_system_instance = None

def get_agent_system():
    global _agent_system_instance
    if _agent_system_instance is None:
        from src.agents.main_agent import EnhancedDatabaseAgentSystem
        _agent_system_instance = EnhancedDatabaseAgentSystem()
        # Optionally connect by default if envs are present
        _agent_system_instance.connect_database('postgresql')
    return _agent_system_instance

# Request Models
class QueryRequest(BaseModel):
    question: str
    session_id: str = "api-session"
    db_type: Optional[str] = None

class PlanRequest(BaseModel):
    question: str
    session_id: str = "api-session"

class SQLRequest(BaseModel):
    question: str
    session_id: str = "api-session"

class ConversationRequest(BaseModel):
    question: str
    session_id: str = "api-session"
    db_type: Optional[str] = None

# Response Models
class ConversationContext(BaseModel):
    previous_question: Optional[str] = None
    previous_result: Optional[Dict] = None
    suggested_follow_up: Optional[str] = None
    context_type: str = "new"  # "new", "follow_up", "clarification"

class QueryResponse(BaseModel):
    response: str
    sql: str
    context: ConversationContext
    suggestions: List[str]
    execution_time: float

# Conversation History Storage (in-memory for now)
conversation_history: Dict[str, List[Dict]] = {}

def analyze_conversation_context(session_id: str, current_question: str) -> ConversationContext:
    """Analyze if the current question is a follow-up to previous conversation"""
    if session_id not in conversation_history or not conversation_history[session_id]:
        return ConversationContext(context_type="new")
    
    last_conversation = conversation_history[session_id][-1]
    last_question = last_conversation.get('question', '')
    last_result = last_conversation.get('result', {})
    
    # Simple follow-up detection
    follow_up_indicators = ['có', 'yes', 'ok', 'được', 'tiếp', 'more', 'thêm', 'and', 'và']
    clarification_indicators = ['gì', 'what', 'nào', 'which', 'sao', 'why', 'tại sao']
    
    current_lower = current_question.lower().strip()
    
    # Check if it's a simple affirmation/continuation
    if current_lower in follow_up_indicators:
        return ConversationContext(
            previous_question=last_question,
            previous_result=last_result,
            suggested_follow_up="continue_previous",
            context_type="follow_up"
        )
    
    # Check if it's asking for clarification
    if any(indicator in current_lower for indicator in clarification_indicators):
        return ConversationContext(
            previous_question=last_question,
            previous_result=last_result,
            suggested_follow_up="clarify_previous",
            context_type="clarification"
        )
    
    # Check if it's a new question but related to previous context
    if last_result and 'rows' in last_result and last_result['rows']:
        # If previous result had data, suggest analysis
        return ConversationContext(
            previous_question=last_question,
            previous_result=last_result,
            suggested_follow_up="analyze_previous_data",
            context_type="related"
        )
    
    return ConversationContext(context_type="new")

def process_follow_up_question(agent, context: ConversationContext, current_question: str) -> str:
    """Process follow-up questions based on conversation context"""
    if context.context_type == "follow_up" and context.previous_question:
        # User said "có" or similar - continue with previous suggestion
        if "phân tích hành vi khách hàng" in context.previous_question.lower():
            # Analyze customer behavior from previous results
            return "Phân tích hành vi khách hàng dựa trên kết quả trước đó"
        elif "tìm kiếm" in context.previous_question.lower():
            # Continue searching with more specific criteria
            return "Tìm kiếm chi tiết hơn về kết quả trước đó"
        else:
            # Generic continuation
            return f"Tiếp tục phân tích: {context.previous_question}"
    
    elif context.context_type == "clarification":
        # User is asking for clarification
        return f"Làm rõ thêm về: {context.previous_question}"
    
    elif context.context_type == "related" and context.previous_result:
        # New question but related to previous data
        if 'rows' in context.previous_result and context.previous_result['rows']:
            # Suggest analysis of the data we found
            return f"Phân tích dữ liệu {len(context.previous_result['rows'])} kết quả từ câu hỏi trước"
    
    return current_question

@app.get("/status")
def status():
    """Get system status"""
    try:
        agent = get_agent_system()
        return {
            "status": "running",
            "database_connected": agent.current_adapter is not None,
            "database_type": agent.current_db_type,
            "conversation_sessions": len(conversation_history),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plan")
def plan(req: PlanRequest):
    """Create query execution plan"""
    try:
        agent = get_agent_system()
        schema = agent.analyze_database_smart()
        context = agent.enhanced_context_agent(req.question, schema)
        plan = agent.planning_agent.create_query_plan(req.question, context, schema)
        return agent.planning_agent.get_plan_summary(plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_sql")
def generate_sql(req: SQLRequest):
    """Generate SQL from natural language question"""
    try:
        agent = get_agent_system()
        schema = agent.analyze_database_smart()
        context = agent.enhanced_context_agent(req.question, schema)
        sql = agent.intelligent_sql_generation(req.question, context, schema)
        return {"sql": sql, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
def process(req: QueryRequest):
    """Process a natural language query"""
    try:
        agent = get_agent_system()
        
        # Connect to database if specified
        if req.db_type:
            agent.connect_database(req.db_type)
        
        # Process the query
        response = agent.process_query(req.question, session_id=req.session_id)
        
        # Get suggestions from conversation agent
        suggestions = agent.conversation_agent.get_smart_suggestions(req.session_id)
        
        # Store in conversation history
        if req.session_id not in conversation_history:
            conversation_history[req.session_id] = []
        
        conversation_history[req.session_id].append({
            'timestamp': time.time(),
            'question': req.question,
            'response': response,
            'session_id': req.session_id
        })
        
        return {
            "response": response,
            "suggestions": suggestions,
            "session_id": req.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation")
def conversation(req: ConversationRequest):
    """Process conversation with context awareness"""
    try:
        agent = get_agent_system()
        
        # Connect to database if specified
        if req.db_type:
            agent.connect_database(req.db_type)
        
        # Analyze conversation context
        conv_context = analyze_conversation_context(req.session_id, req.question)
        
        # Process follow-up questions
        processed_question = process_follow_up_question(agent, conv_context, req.question)
        
        # Process the query with context
        start_time = time.time()
        response = agent.process_query(processed_question, session_id=req.session_id)
        execution_time = time.time() - start_time
        
        # Get suggestions
        suggestions = agent.conversation_agent.get_smart_suggestions(req.session_id)
        
        # Store in conversation history
        if req.session_id not in conversation_history:
            conversation_history[req.session_id] = []
        
        # Get the actual result from the response
        result = None
        if "Found" in response and "results:" in response:
            # Extract result info from response
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if "Found" in line and "results:" in line:
                    # Try to get the actual data rows
                    if i + 2 < len(lines):
                        result = {
                            'row_count': int(line.split()[1]),
                            'rows': lines[i+2:i+12] if i+12 < len(lines) else lines[i+2:]
                        }
                    break
        
        conversation_history[req.session_id].append({
            'timestamp': time.time(),
            'question': req.question,
            'processed_question': processed_question,
            'response': response,
            'result': result,
            'context': conv_context.dict(),
            'session_id': req.session_id
        })
        
        return QueryResponse(
            response=response,
            sql="",  # We don't have direct access to SQL here
            context=conv_context,
            suggestions=suggestions,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
def verify(req: SQLRequest):
    """Verify query results against business rules"""
    try:
        agent = get_agent_system()
        schema = agent.analyze_database_smart()
        context = agent.enhanced_context_agent(req.question, schema)
        
        # Generate and execute SQL
        sql = agent.intelligent_sql_generation(req.question, context, schema)
        result, error = agent.current_adapter.execute_query(sql)
        
        if error:
            return {"error": error}
        
        # Verify results
        validations = agent.verification_agent.verify_query_result(result, req.question, context, schema)
        summary = agent.verification_agent.get_validation_summary(validations)
        
        return {
            "summary": summary, 
            "validations": [v.__dict__ for v in validations],
            "sql": sql,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{session_id}")
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversation_history:
        return {"conversations": []}
    
    return {
        "session_id": session_id,
        "conversations": conversation_history[session_id],
        "total_turns": len(conversation_history[session_id])
    }

@app.delete("/conversation/{session_id}")
def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_history:
        del conversation_history[session_id]
    return {"message": f"Conversation history cleared for session {session_id}"}

@app.get("/conversation/{session_id}/suggestions")
def get_suggestions(session_id: str):
    """Get smart suggestions for a conversation session"""
    try:
        agent = get_agent_system()
        suggestions = agent.conversation_agent.get_smart_suggestions(session_id)
        return {"suggestions": suggestions, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run():
    """Run the MCP server"""
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))

if __name__ == "__main__":
    run()