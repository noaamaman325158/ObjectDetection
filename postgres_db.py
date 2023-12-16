import psycopg2
from psycopg2 import sql
# Database connection parameters
db_name = "postgres"  # default database name
db_user = "postgres"  # default username
db_pass = "mysecretpassword"  # the password you set
db_host = "localhost"  # or use the Docker machine IP if you're running on Docker Toolbox
db_port = "5432"  # the port you mapped

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    database=db_name,
    user=db_user,
    password=db_pass,
    host=db_host,
    port=db_port
)

# Create a cursor object
cur = conn.cursor()

# SQL to create a table
create_table_query = sql.SQL("""
    CREATE TABLE IF NOT EXISTS object_detections (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        object_type VARCHAR(50),
        confidence_score REAL,
        location VARCHAR(100),
        bounding_box TEXT,
        image_snapshot BYTEA,
        additional_metadata JSONB
    )
""")

# Execute the query
cur.execute(create_table_query)

# Commit the changes
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()
