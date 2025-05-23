import psycopg2
from psycopg2 import sql
import sys

# Database configuration
DB_CONFIG = {
    'dbname': 'algodb',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': '5432',
}

def check_database_connection():
    try:
        # Try to connect to the database
        conn = psycopg2.connect(
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        print(f"Successfully connected to database {DB_CONFIG['dbname']}")
        
        # Check if the required tables exist
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = [table[0] for table in cursor.fetchall()]
        print(f"Available tables: {', '.join(tables)}")
        
        # Check for specific tables
        required_tables = ['ariane_place_sorted', 'ariane_cts_sorted', 'ariane_route_sorted']
        for table in required_tables:
            if table in tables:
                # Count rows in the table
                cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
                count = cursor.fetchone()[0]
                print(f"Table {table} exists with {count} rows")
            else:
                print(f"Table {table} does not exist")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_database_connection()
    sys.exit(0 if success else 1)