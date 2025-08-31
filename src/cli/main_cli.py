"""
Main CLI interface for the database agent system
"""

from ..agents.main_agent import EnhancedDatabaseAgentSystem
from ..database.models import DatabaseSchema


def main():
    """Enhanced CLI interface"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 Multi-Database Intelligent Agent System 🤖        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    agent = EnhancedDatabaseAgentSystem()
    
    # Database selection
    print("\n📊 Available Database Types:")
    print("1. PostgreSQL")
    print("2. MySQL")
    print("3. SQLite (coming soon)")
    print("4. MongoDB (coming soon)")
    
    db_choice = input("\nSelect database type (1-2): ").strip()
    
    db_type_map = {
        '1': 'postgresql',
        '2': 'mysql'
    }
    
    selected_db = db_type_map.get(db_choice, 'postgresql')
    
    # Connect to database
    print(f"\n🔌 Connecting to {selected_db}...")
    if not agent.connect_database(selected_db):
        print("Failed to connect to database. Please check your configuration.")
        return
    
    # Analyze schema
    print("\n🔍 Analyzing database schema...")
    schema = agent.analyze_database_smart()
    
    print(f"\n✅ Connected to {selected_db} database")
    print(f"📊 Found {len(schema.tables)} tables")
    print(f"🔗 Found {len(schema.relationships)} relationships")
    
    # Show available tables
    print("\n📋 Available tables:")
    for i, table_name in enumerate(list(schema.tables.keys())[:10], 1):
        table = schema.tables[table_name]
        print(f"  {i}. {table_name} ({table.row_count} rows, {len(table.columns)} columns)")
    
    if len(schema.tables) > 10:
        print(f"  ... and {len(schema.tables) - 10} more tables")
    
    print("\n" + "="*60)
    print("💡 Commands:")
    print("  - Type your question in natural language")
    print("  - 'SCHEMA <table>' - Show table schema")
    print("  - 'RELATIONSHIPS' - Show table relationships")
    print("  - 'SWITCH DB' - Switch database type")
    print("  - 'CLEAR' - Clear context history")
    print("  - 'EXIT' - Exit the system")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.upper() == 'EXIT':
                print("\n👋 Goodbye!")
                break
                
            elif user_input.upper().startswith('SCHEMA'):
                parts = user_input.split()
                if len(parts) > 1:
                    table_name = parts[1]
                    if table_name in schema.tables:
                        table = schema.tables[table_name]
                        print(f"\n📋 Schema for {table_name}:")
                        print(f"Rows: {table.row_count}")
                        print("\nColumns:")
                        for col in table.columns:
                            print(f"  - {col['name']}: {col['type']} {'NOT NULL' if not col['nullable'] else 'NULL'}")
                        if table.foreign_keys:
                            print("\nForeign Keys:")
                            for fk in table.foreign_keys:
                                print(f"  - {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}")
                    else:
                        print(f"Table '{table_name}' not found")
                else:
                    print("Usage: SCHEMA <table_name>")
            
            elif user_input.upper() == 'RELATIONSHIPS':
                print("\n🔗 Table Relationships:")
                for rel in schema.relationships[:20]:
                    rel_type = rel.get('type', 'foreign_key')
                    confidence = f" (confidence: {rel.get('confidence', 1.0):.1%})" if 'confidence' in rel else ""
                    print(f"  {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']} [{rel_type}]{confidence}")
                if len(schema.relationships) > 20:
                    print(f"  ... and {len(schema.relationships) - 20} more relationships")
            
            elif user_input.upper() == 'SWITCH DB':
                print("\nSelect new database type:")
                print("1. PostgreSQL")
                print("2. MySQL")
                db_choice = input("Choice (1-2): ").strip()
                new_db = db_type_map.get(db_choice, 'postgresql')
                if agent.connect_database(new_db):
                    schema = agent.analyze_database_smart()
                    print(f"✅ Switched to {new_db}")
            
            elif user_input.upper() == 'CLEAR':
                agent.context_history.clear()
                agent.schema_cache.clear()
                print("🗑️ Context and cache cleared")
            
            else:
                # Process natural language query
                response = agent.process_query(user_input)
                print("\n" + "="*60)
                print(response)
                print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'EXIT' to quit")


if __name__ == "__main__":
    main()
