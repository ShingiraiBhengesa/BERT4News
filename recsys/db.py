"""Database operations for the news recommendation system."""
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta

from .config import DATABASE_PATH, TOPICS

logger = logging.getLogger(__name__)


class NewsDatabase:
    """Handle all database operations for the news recommendation system."""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    article_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT,
                    content TEXT,
                    authors TEXT,
                    source TEXT NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    topics TEXT,  -- JSON array of topics
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    signup_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    declared_topics TEXT,  -- JSON array of preferred topics
                    profile_data TEXT  -- JSON for additional profile info
                )
            """)
            
            # Interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    article_id INTEGER NOT NULL,
                    event TEXT NOT NULL,  -- 'click', 'like', 'dislike', 'read', 'share'
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dwell_time_s INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (article_id) REFERENCES articles (article_id)
                )
            """)
            
            # User preferences (learned from interactions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    affinity_score REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, topic),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Article embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    article_id INTEGER PRIMARY KEY,
                    embedding_type TEXT NOT NULL,  -- 'tfidf', 'sentence_transformer'
                    embedding BLOB NOT NULL,  -- Serialized numpy array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES articles (article_id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_user_id 
                ON interactions(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_article_id 
                ON interactions(article_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_published_at 
                ON articles(published_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_source 
                ON articles(source)
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def insert_articles(self, articles_df: pd.DataFrame) -> int:
        """Insert articles from DataFrame into database."""
        with self.get_connection() as conn:
            count = articles_df.to_sql(
                'articles', conn, if_exists='append', index=False
            )
            logger.info(f"Inserted {count} articles")
            return count
    
    def insert_interactions(self, interactions_df: pd.DataFrame) -> int:
        """Insert interactions from DataFrame into database."""
        with self.get_connection() as conn:
            count = interactions_df.to_sql(
                'interactions', conn, if_exists='append', index=False
            )
            logger.info(f"Inserted {count} interactions")
            return count
    
    def insert_users(self, users_df: pd.DataFrame) -> int:
        """Insert users from DataFrame into database."""
        with self.get_connection() as conn:
            count = users_df.to_sql(
                'users', conn, if_exists='append', index=False
            )
            logger.info(f"Inserted {count} users")
            return count
    
    def get_user_interactions(self, user_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all interactions for a specific user."""
        query = """
            SELECT i.*, a.title, a.topics, a.source
            FROM interactions i
            JOIN articles a ON i.article_id = a.article_id
            WHERE i.user_id = ?
            ORDER BY i.ts DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[user_id])
    
    def get_article_interactions(self, article_id: int) -> pd.DataFrame:
        """Get all interactions for a specific article."""
        query = """
            SELECT * FROM interactions
            WHERE article_id = ?
            ORDER BY ts DESC
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[article_id])
    
    def get_user_item_matrix(self, 
                           interaction_types: List[str] = None,
                           min_interactions: int = 5) -> pd.DataFrame:
        """Get user-item interaction matrix for collaborative filtering."""
        if interaction_types is None:
            interaction_types = ['click', 'like', 'read']
        
        placeholders = ','.join(['?' for _ in interaction_types])
        query = f"""
            SELECT user_id, article_id, COUNT(*) as interaction_count
            FROM interactions
            WHERE event IN ({placeholders})
            GROUP BY user_id, article_id
            HAVING interaction_count >= ?
        """
        
        params = interaction_types + [min_interactions]
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_recent_articles(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """Get recent articles for content-based recommendations."""
        query = """
            SELECT * FROM articles
            WHERE published_at >= date('now', '-{} day')
            ORDER BY published_at DESC
            LIMIT ?
        """.format(days)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[limit])
    
    def get_popular_articles(self, days: int = 7, limit: int = 100) -> pd.DataFrame:
        """Get popular articles based on interaction count."""
        query = """
            SELECT a.*, COUNT(i.article_id) as interaction_count
            FROM articles a
            JOIN interactions i ON a.article_id = i.article_id
            WHERE a.published_at >= date('now', '-{} day')
            GROUP BY a.article_id
            ORDER BY interaction_count DESC
            LIMIT ?
        """.format(days)
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=[limit])
    
    def get_user_topic_affinity(self, user_id: int) -> Dict[str, float]:
        """Calculate user's topic affinity based on interaction history."""
        query = """
            SELECT a.topics, COUNT(*) as interaction_count
            FROM interactions i
            JOIN articles a ON i.article_id = a.article_id
            WHERE i.user_id = ? AND i.event IN ('click', 'like', 'read')
            GROUP BY a.topics
        """
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[user_id])
            
        # Parse topics and calculate affinity scores
        topic_counts = {}
        total_interactions = 0
        
        for _, row in df.iterrows():
            if row['topics']:
                # Assuming topics are stored as comma-separated values
                topics = [t.strip() for t in row['topics'].split(',')]
                count = row['interaction_count']
                total_interactions += count
                
                for topic in topics:
                    if topic in TOPICS:  # Only consider predefined topics
                        topic_counts[topic] = topic_counts.get(topic, 0) + count
        
        # Normalize to get affinity scores
        if total_interactions > 0:
            return {topic: count / total_interactions 
                   for topic, count in topic_counts.items()}
        return {}
    
    def execute_sql_queries(self) -> Dict[str, pd.DataFrame]:
        """Execute the 5+ SQL queries mentioned in the requirements."""
        queries = {
            "top_readers": """
                SELECT user_id, COUNT(*) AS interaction_count
                FROM interactions
                GROUP BY user_id
                ORDER BY interaction_count DESC
                LIMIT 20
            """,
            
            "article_popularity": """
                SELECT article_id, title, COUNT(i.article_id) AS interaction_count
                FROM articles a
                JOIN interactions i ON a.article_id = i.article_id
                GROUP BY a.article_id, a.title
                ORDER BY interaction_count DESC
                LIMIT 20
            """,
            
            "user_topic_affinity": """
                SELECT i.user_id, a.topics, COUNT(*) AS interaction_count
                FROM interactions i
                JOIN articles a ON i.article_id = a.article_id
                WHERE a.topics IS NOT NULL
                GROUP BY i.user_id, a.topics
                ORDER BY i.user_id, interaction_count DESC
            """,
            
            "recent_quality_sources": """
                SELECT source, COUNT(*) AS article_count
                FROM articles
                WHERE published_at >= date('now', '-14 day')
                GROUP BY source
                ORDER BY article_count DESC
            """,
            
            "cold_start_popular": """
                SELECT a.article_id, a.title, a.source, COUNT(i.article_id) AS popularity
                FROM articles a
                LEFT JOIN interactions i ON a.article_id = i.article_id
                WHERE a.published_at >= date('now', '-7 day')
                GROUP BY a.article_id, a.title, a.source
                ORDER BY popularity DESC, a.published_at DESC
                LIMIT 200
            """,
            
            "user_engagement_patterns": """
                SELECT 
                    event,
                    COUNT(*) AS event_count,
                    AVG(dwell_time_s) AS avg_dwell_time
                FROM interactions
                WHERE dwell_time_s > 0
                GROUP BY event
                ORDER BY event_count DESC
            """
        }
        
        results = {}
        with self.get_connection() as conn:
            for query_name, query in queries.items():
                try:
                    results[query_name] = pd.read_sql_query(query, conn)
                    logger.info(f"Executed query: {query_name}")
                except Exception as e:
                    logger.error(f"Error executing query {query_name}: {e}")
                    results[query_name] = pd.DataFrame()
        
        return results
