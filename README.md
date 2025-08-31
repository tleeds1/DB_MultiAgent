# Multi-Database Intelligent Agent System

A smart database query system that supports multiple database types with intelligent constraint-aware operations.

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
│   │   └── main_agent.py          # Main agent system coordinator
│   ├── utils/                     # Utility classes
│   │   ├── __init__.py
│   │   ├── query_optimizer.py     # SQL query optimization
│   │   ├── notification.py        # Telegram/Slack notifications
│   │   └── schema_analyzer.py     # Schema analysis and relationship detection
│   └── cli/                       # Command line interface
│       ├── __init__.py
│       └── main_cli.py            # Main CLI interface
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── README_STRUCTURED.md          # This file
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

### 📱 **Notification System**
- **Telegram Integration** - Comprehensive reports for every operation
- **Slack Support** - Team collaboration notifications
- **Structured Reports** - Detailed analysis of database changes

### ⚡ **Performance & Optimization**
- **Schema Caching** - Intelligent caching with TTL
- **Query Optimization** - Automatic SQL optimization
- **Index Hints** - Database-specific performance hints

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DB_MCP
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
python main.py
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

# Notification Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK=your_slack_webhook_url
```

## 💡 Usage Examples

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

## 🧪 **Testing**

```bash
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
