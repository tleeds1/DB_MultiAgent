# Multi-Database Intelligent Agent System

A smart database query system that supports multiple database types with intelligent constraint-aware operations, featuring a powerful MCP (Model Context Protocol) server for API-based interactions.

## 🏗️ Project Structure

```
DB_MCP/
├── src/
│   ├── __init__.py                 # Main package
│   ├── database/                   # Database adapters and models
│   │   ├── __init__.py
│   │   ├── models.py              # Data models (TableInfo, DatabaseSchema)
│   │   ├── adapters.py            # Database adapters (PostgreSQL, MySQL)
│   │   └── factory.py             # Database factory for creating adapters
│   ├── agents/                    # Intelligent agents
│   │   ├── __init__.py
│   │   ├── context_agent.py       # Context analysis and language detection
│   │   ├── sql_generation_agent.py # SQL generation with LLM
│   │   ├── execution_agent.py     # Query execution and monitoring
│   │   ├── planning_agent.py      # Query planning and optimization
│   │   ├── verification_agent.py  # Result verification and business rules
│   │   ├── conversation_agent.py  # Conversation context and suggestions
│   │   └── main_agent.py          # Main agent system coordinator
│   ├── utils/                     # Utility classes
│   │   ├── __init__.py
│   │   ├── query_optimizer.py     # SQL query optimization
│   │   ├── notification.py        # Telegram/Slack/WhatsApp notifications
│   │   ├── schema_analyzer.py     # Schema analysis and relationship detection
│   │   └── llm_monitor.py         # LLM usage monitoring and fallback
│   └── cli/                       # Command line interface
│       ├── __init__.py
│       └── main_cli.py            # Main CLI interface
├── server.py                      # MCP Server (FastAPI)
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🚀 Features

### 🔍 **Multi-Database Support**
- **PostgreSQL** - Full support with advanced schema inspection
- **MySQL** - Complete adapter with relationship detection
- **SQLite** - Coming soon
- **MongoDB** - Coming soon

### 🧠 **Intelligent Operations**
- **Constraint-Aware Insertions** - Automatically determines correct order based on foreign key relationships
- **Dynamic Schema Analysis** - No hardcoded table relationships
- **Smart Context Detection** - Automatically detects Vietnamese/English and query complexity
- **Relationship Graph Analysis** - Uses NetworkX for dependency analysis

### 🤖 **AI-Powered Features**
- **LLM Integration** - Google Gemini with Ollama Llama fallback
- **Conversation Memory** - Maintains context across multiple queries
- **Follow-up Question Handling** - Intelligently processes "có", "yes", "ok" responses
- **Smart Suggestions** - Context-aware recommendations for next queries

### 📱 **Notification System**
- **Telegram Integration** - Comprehensive reports for every operation
- **Slack Support** - Team collaboration notifications
- **WhatsApp Support** - Twilio-based messaging (optional)
- **Structured Reports** - Detailed analysis of database changes

### ⚡ **Performance & Optimization**
- **Schema Caching** - Intelligent caching with TTL
- **Query Optimization** - Automatic SQL optimization
- **Index Hints** - Database-specific performance hints
- **LLM Monitoring** - Usage tracking and automatic model switching

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/tleeds1/DB_MultiAgent
cd DB_MultiAgent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your database and API credentials
```

4. **Run the system**
```bash
# CLI Mode
python main.py

# MCP Server Mode
python server.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DB=your_database

# LLM Configuration
GEMINI_API=your_gemini_api_key

# Ollama Configuration (for Llama fallback)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# Notification Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK=your_slack_webhook_url
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_phone
```

## 🌐 **MCP Server Usage**

The MCP server provides a RESTful API for programmatic access to the intelligent database agent system.

### **Starting the Server**

```bash
# Start the server
python server.py

# Or using uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8080
```

The server will be available at `http://localhost:8080`

### **API Endpoints**

#### **1. System Status**
```bash
GET /status
```
Returns system health and connection status.

#### **2. Query Planning**
```bash
POST /plan
Content-Type: application/json

{
  "question": "Find top 10 customers by revenue",
  "session_id": "user-123"
}
```
Creates an execution plan for complex queries.

#### **3. SQL Generation**
```bash
POST /generate_sql
Content-Type: application/json

{
  "question": "Show me customers from region 82",
  "session_id": "user-123"
}
```
Generates SQL from natural language without execution.

#### **4. Query Processing**
```bash
POST /process
Content-Type: application/json

{
  "question": "Find customers in Vietnam region 82",
  "session_id": "user-123",
  "db_type": "postgresql"
}
```
Processes natural language queries and returns results.

#### **5. Conversation API (Recommended)**
```bash
POST /conversation
Content-Type: application/json

{
  "question": "Tìm khách hàng vùng 82",
  "session_id": "user-123"
}
```
**Most powerful endpoint** - Handles conversation context and follow-up questions.

#### **6. Result Verification**
```bash
POST /verify
Content-Type: application/json

{
  "question": "Verify customer data integrity",
  "session_id": "user-123"
}
```
Verifies query results against business rules.

#### **7. Conversation Management**
```bash
# Get conversation history
GET /conversation/{session_id}

# Get smart suggestions
GET /conversation/{session_id}/suggestions

# Clear conversation history
DELETE /conversation/{session_id}
```

### **🎯 Follow-up Question Handling**

The MCP server intelligently handles follow-up questions:

#### **Example 1: Customer Analysis**
```bash
# First query
POST /conversation
{
  "question": "Tìm khách hàng vùng 82",
  "session_id": "session-1"
}

# Response: Found 3 customers + suggestions
# Agent suggests: "Bạn có muốn phân tích hành vi khách hàng không?"

# Follow-up query
POST /conversation
{
  "question": "có",
  "session_id": "session-1"
}

# Agent understands "có" means "continue with previous suggestion"
# Generates: "Phân tích hành vi khách hàng dựa trên kết quả tìm được trước đó"
# Returns detailed customer behavior analysis
```

#### **Example 2: Data Exploration**
```bash
# First query
POST /conversation
{
  "question": "Show me recent orders",
  "session_id": "session-2"
}

# Follow-up
POST /conversation
{
  "question": "yes",
  "session_id": "session-2"
}

# Agent continues with order analysis
```

#### **Supported Follow-up Indicators**
- **Vietnamese**: "có", "được", "tiếp", "thêm", "và"
- **English**: "yes", "ok", "okay", "more", "and"
- **Clarification**: "gì", "what", "nào", "which", "sao", "why", "tại sao"

### **🔧 Testing the MCP Server**

#### **Using PowerShell**
```powershell
# Test status
Invoke-RestMethod -Uri "http://localhost:8080/status" -Method Get

# Test conversation
$body = @{
    question = "Tim khach hang vung 82"
    session_id = "test-1"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/conversation" -Method Post -Body $body -ContentType "application/json"

# Test follow-up
$followup = @{
    question = "co"
    session_id = "test-1"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/conversation" -Method Post -Body $followup -ContentType "application/json"
```

#### **Using cURL**
```bash
# Test conversation
curl -X POST "http://localhost:8080/conversation" \
  -H "Content-Type: application/json" \
  -d '{"question": "Find customers", "session_id": "test-1"}'

# Test follow-up
curl -X POST "http://localhost:8080/conversation" \
  -H "Content-Type: application/json" \
  -d '{"question": "yes", "session_id": "test-1"}'
```

#### **Using Python Requests**
```python
import requests

# Test conversation
response = requests.post("http://localhost:8080/conversation", json={
    "question": "Find customers in region 82",
    "session_id": "python-test"
})

# Test follow-up
followup = requests.post("http://localhost:8080/conversation", json={
    "question": "co",
    "session_id": "python-test"
})

print(followup.json())
```

## 💡 **CLI Usage Examples**

### 🔍 **Schema Analysis**
```bash
# View table relationships
RELATIONSHIPS

# View specific table schema
SCHEMA customers
```

### 🗣️ **Natural Language Queries**

#### **Simple Queries**
```
💬 Your question: Show me all customers
💬 Your question: Count total orders
💬 Your question: Find customers in region 82
```

#### **Complex Queries**
```
💬 Your question: Show top 10 customers by order value with their addresses
💬 Your question: Find average order value per region
💬 Your question: List customers who haven't ordered in the last 30 days
```

#### **Write Operations**
```
💬 Your question: Add a new customer named "John Doe" with email "john@example.com"
💬 Your question: Update customer ID 123 to have status "active"
💬 Your question: Delete all inactive customers
```

## 🔗 **Smart Constraint Handling**

The system automatically handles foreign key constraints:

### **Example: Adding Customer with Address**
```
💬 Your question: Add customer "Jane Smith" with address "123 Main St, City, Country"
```

**System automatically:**
1. **Analyzes dependencies** - Customer table depends on Address table
2. **Determines order** - Insert address first, then customer
3. **Generates SQL** - Uses CTEs or multiple statements in correct order
4. **Executes safely** - No constraint violations

### **Generated SQL Example:**
```sql
WITH new_address AS (
    INSERT INTO addresses (street, city, country) 
    VALUES ('123 Main St', 'City', 'Country') 
    RETURNING addressid
)
INSERT INTO customers (name, addressid) 
SELECT 'Jane Smith', addressid FROM new_address;
```

## 🏗️ **Architecture Benefits**

### **Modular Design**
- **Easy to extend** - Add new database types by implementing `DatabaseAdapter`
- **Separation of concerns** - Each agent has a specific responsibility
- **Testable components** - Each module can be tested independently

### **Scalable Structure**
- **Plugin architecture** - Easy to add new notification channels
- **Configurable agents** - Adjust behavior without code changes
- **Cache management** - Intelligent caching reduces database load

### **Maintainable Code**
- **Clear interfaces** - Abstract base classes define contracts
- **Documentation** - Comprehensive docstrings and type hints
- **Error handling** - Graceful fallbacks and detailed error messages

### **AI-Powered Intelligence**
- **Conversation Memory** - Maintains context across sessions
- **Follow-up Understanding** - Processes natural language continuations
- **Smart Suggestions** - Context-aware recommendations
- **LLM Fallback** - Automatic model switching for reliability

## 🔮 **Future Enhancements**

### **Planned Features**
- **SQLite Support** - Lightweight database adapter
- **MongoDB Support** - NoSQL database integration
- **Advanced Caching** - Redis-based distributed caching
- **Query History** - Persistent query storage and analysis
- **Performance Metrics** - Detailed query performance tracking

### **AI Enhancements**
- **Query Learning** - Learn from user corrections
- **Schema Evolution** - Handle schema changes automatically
- **Natural Language Training** - Improve query understanding over time
- **Multi-language Support** - Enhanced Vietnamese/English processing

## 🧪 **Testing**

```bash
# Test MCP server
python -m py_compile server.py
python -c "import server; print('✅ Server imports successfully')"

# Test agents
python -m py_compile src/agents/main_agent.py
python -m py_compile src/agents/conversation_agent.py

# Run tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📊 **Monitoring & Logging**

The system provides comprehensive monitoring:

- **Query Performance** - Execution time tracking
- **Error Rates** - Failed query monitoring
- **Usage Statistics** - Query count and complexity analysis
- **Database Health** - Connection status and performance metrics
- **LLM Usage** - API calls, tokens, and cost tracking
- **Conversation Analytics** - Session length and user engagement

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch** - `git checkout -b feature/amazing-feature`
3. **Make your changes** - Follow the existing code structure
4. **Add tests** - Ensure new features are tested
5. **Submit a pull request** - Describe your changes clearly

## 📝 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 **Support**

- **Documentation** - Check this README and code comments
- **Issues** - Report bugs via GitHub Issues
- **Discussions** - Ask questions in GitHub Discussions

---

**Author: tleeds1 - Le Doan Tho - HCMUT**
