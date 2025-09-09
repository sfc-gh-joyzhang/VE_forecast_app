import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def set_connection(role_name=None, warehouse_name=None):
    """
    Create Snowflake connection using browser authentication
    Uses environment variables or defaults for connection parameters.
    
    Environment Variables:
    - SNOWFLAKE_USER: Your Snowflake username
    - SNOWFLAKE_ACCOUNT: Your Snowflake account identifier  
    - SNOWFLAKE_ROLE: Default role to use
    - SNOWFLAKE_WAREHOUSE: Default warehouse to use
    
    Args:
        role_name: Override role (optional)
        warehouse_name: Override warehouse (optional)
    
    Returns:
        tuple: (cursor, connection) for executing queries
    """
    # Get connection parameters from environment or use working defaults
    user = os.getenv('SNOWFLAKE_USER', 'jozhang')  # Default for testing
    account = os.getenv('SNOWFLAKE_ACCOUNT', 'snowhouse')  # Default for testing
    default_role = os.getenv('SNOWFLAKE_ROLE', 'PUBLIC')  # Use PUBLIC (confirmed working)
    default_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'SNOWADHOC')  # Use SNOWADHOC (confirmed working)
    
    # Use provided parameters or fall back to defaults
    final_role = role_name or default_role
    final_warehouse = warehouse_name or default_warehouse
    
    try:
        conn = snowflake.connector.connect(
            user=user,
            account=account,
            authenticator='externalbrowser'
        )
        
        cursor = conn.cursor()
        
        # Set basic session parameters
        cursor.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
        
        # Set role and warehouse (now using confirmed working values)
        if final_role:
            cursor.execute(f"USE ROLE {final_role}")
            
        if final_warehouse:
            cursor.execute(f"USE WAREHOUSE {final_warehouse}")
            
        return cursor, conn
        
    except Exception as e:
        error_msg = f"""
Snowflake Connection Failed: {str(e)}

Check these settings:
• SNOWFLAKE_USER: {user}
• SNOWFLAKE_ACCOUNT: {account}  
• SNOWFLAKE_ROLE: {final_role}
• SNOWFLAKE_WAREHOUSE: {final_warehouse}

To fix:
1. Create .env file with your Snowflake credentials
2. Or set environment variables
3. Make sure your account identifier is correct
        """
        raise ConnectionError(error_msg)

def execute_sql(cursor, sql_query):
    """
    Execute SQL query and return results as pandas DataFrame
    
    Args:
        cursor: Snowflake cursor
        sql_query: SQL query string
        
    Returns:
        DataFrame: Query results as pandas DataFrame
    """
    try:
        cursor.execute(sql_query)
        
        # Use fetch_pandas_all() if available (newer connector versions)
        if hasattr(cursor, 'fetch_pandas_all'):
            return cursor.fetch_pandas_all()
        else:
            # Fallback for older connector versions
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame(results, columns=columns)
            
    except Exception as e:
        raise RuntimeError(f"SQL execution failed: {str(e)}\nQuery: {sql_query[:100]}...")

def close_connection(cursor, connection):
    """
    Close Snowflake connection
    
    Args:
        cursor: Snowflake cursor to close
        connection: Snowflake connection to close
    """
    try:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    except:
        pass  # Ignore close errors

# Convenience function for one-off queries
def query_snowflake(sql_query, role_name=None, warehouse_name=None):
    """
    Execute single query with automatic connection management
    
    Args:
        sql_query: SQL query to execute
        role_name: Override role (optional)
        warehouse_name: Override warehouse (optional)
        
    Returns:
        DataFrame: Query results
    """
    cursor, conn = set_connection(role_name, warehouse_name)
    try:
        return execute_sql(cursor, sql_query)
    finally:
        close_connection(cursor, conn)