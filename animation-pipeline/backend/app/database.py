"""
Database configuration and session management for Animation Pipeline 2.0
Handles PostgreSQL connection, session creation, and database initialization
Located in backend/app/database.py
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent  # backend/app -> backend -> project root
env_path = project_root / ".env"
load_dotenv(env_path)

from models.scenes import Base, create_tables

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self):
        # Database connection parameters
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5433')  # Your custom PostgreSQL port
        self.db_name = os.getenv('DB_NAME', 'animation_pipeline')
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', 'postgres')
        
        # Build connection URL
        self.database_url = (
            f"postgresql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_database_tables(self):
        """Create all database tables"""
        print("Creating database tables...")
        try:
            create_tables(self.engine)
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating database tables: {e}")
            raise
    
    def test_connection(self):
        """Test database connection"""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))  # Fixed: wrapped in text()
                print("✅ Database connection successful")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

# Global database configuration instance
db_config = DatabaseConfig()

# Dependency for FastAPI to get database sessions
def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session for FastAPI endpoints
    Ensures proper session cleanup after request
    """
    db = db_config.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    Use this for non-FastAPI database operations
    """
    db = db_config.SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Initialization function
def init_database():
    """Initialize database - test connection and create tables"""
    print("Initializing Animation Pipeline database...")
    
    # Test connection
    if not db_config.test_connection():
        raise Exception("Failed to connect to database")
    
    # Create tables
    db_config.create_database_tables()
    
    print("✅ Database initialization complete")

if __name__ == "__main__":
    # Run database initialization when script is executed directly
    init_database()
