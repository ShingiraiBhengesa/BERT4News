"""Data ingestion script to load news articles and interactions into SQLite database."""
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from recsys.db import NewsDatabase
from recsys.config import RAW_DATA_DIR, TOPICS
from recsys.utils.text import preprocess_articles_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_articles: int = 1000, n_users: int = 200, n_interactions: int = 5000):
    """Generate sample news data for demonstration purposes."""
    logger.info("Generating sample data...")
    
    # Sample article titles and sources
    tech_titles = [
        "AI Revolution Transforms Healthcare Industry",
        "Quantum Computing Breakthrough at Tech Giant", 
        "New Programming Language Gains Developer Adoption",
        "Cybersecurity Threats Rise in Remote Work Era",
        "Blockchain Technology Disrupts Financial Sector"
    ]
    
    politics_titles = [
        "Election Campaign Enters Final Phase",
        "New Policy Initiative Aims to Boost Economy", 
        "International Trade Agreement Reached",
        "Government Announces Infrastructure Plan",
        "Legislative Session Focuses on Climate Action"
    ]
    
    business_titles = [
        "Market Rally Continues Despite Economic Uncertainty",
        "Startup Raises Record Funding Round",
        "Company Reports Strong Quarterly Earnings",
        "Supply Chain Issues Impact Global Trade",
        "Inflation Concerns Influence Federal Policy"
    ]
    
    title_templates = {
        'technology': tech_titles,
        'politics': politics_titles,
        'business': business_titles,
        'health': ["Medical Breakthrough Shows Promise", "New Treatment Options Available"],
        'science': ["Research Reveals Climate Impact", "Space Mission Achieves Milestone"],
        'sports': ["Championship Game Draws Record Audience", "Athlete Breaks Long-Standing Record"],
        'entertainment': ["New Movie Breaks Box Office Records", "Celebrity Announces Major Project"],
        'finance': ["Stock Market Volatility Continues", "Economic Indicators Show Growth"],
        'world': ["International Crisis Requires Urgent Action", "Cultural Exchange Program Launched"],
        'lifestyle': ["Health Trend Gains Popularity", "Travel Restrictions Updated"]
    }
    
    sources = ['reuters', 'bbc', 'cnn', 'nytimes', 'guardian', 'techcrunch', 'bloomberg', 'wsj', 'npr', 'ap']
    
    # Generate articles
    articles_data = []
    for i in range(n_articles):
        topic = np.random.choice(TOPICS)
        title_base = np.random.choice(title_templates[topic])
        
        # Add some variation to titles
        variations = ["Updated", "Breaking", "Analysis", "Report", "Study Shows"]
        if np.random.random() < 0.3:  # 30% chance of variation
            title = f"{np.random.choice(variations)}: {title_base}"
        else:
            title = title_base
        
        summary = f"This is a summary for the article about {topic.lower()}. " + \
                 "The article provides detailed information and analysis on this important topic."
        
        content = summary + " " + "This would be the full article content with detailed information, " + \
                 "quotes from experts, background context, and comprehensive analysis of the topic. " * 3
        
        # Random publication date within last 60 days
        days_ago = np.random.randint(0, 60)
        pub_date = datetime.now() - timedelta(days=days_ago)
        
        article = {
            'article_id': i + 1,
            'title': title,
            'summary': summary,
            'content': content,
            'authors': f"Reporter {np.random.randint(1, 50)}",
            'source': np.random.choice(sources),
            'published_at': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
            'topics': topic,
            'url': f"https://example.com/article/{i+1}"
        }
        articles_data.append(article)
    
    articles_df = pd.DataFrame(articles_data)
    
    # Generate users
    users_data = []
    for i in range(n_users):
        # Random topic preferences
        n_topics = np.random.randint(1, 4)  # 1-3 preferred topics
        preferred_topics = np.random.choice(TOPICS, size=n_topics, replace=False).tolist()
        
        user = {
            'user_id': i + 1,
            'signup_ts': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d %H:%M:%S'),
            'declared_topics': ','.join(preferred_topics),
            'profile_data': json.dumps({'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'])})
        }
        users_data.append(user)
    
    users_df = pd.DataFrame(users_data)
    
    # Generate interactions
    interactions_data = []
    events = ['click', 'read', 'like', 'share', 'dislike']
    event_weights = [0.5, 0.25, 0.15, 0.05, 0.05]  # Click is most common
    
    for i in range(n_interactions):
        user_id = np.random.randint(1, n_users + 1)
        article_id = np.random.randint(1, n_articles + 1)
        event = np.random.choice(events, p=event_weights)
        
        # Simulate realistic dwell times
        if event == 'read':
            dwell_time = np.random.randint(30, 300)  # 30 seconds to 5 minutes
        elif event == 'click':
            dwell_time = np.random.randint(5, 60)    # 5 seconds to 1 minute
        else:
            dwell_time = np.random.randint(1, 30)    # Quick interactions
        
        # Random timestamp within last 30 days
        days_ago = np.random.randint(0, 30)
        hours_ago = np.random.randint(0, 24)
        minutes_ago = np.random.randint(0, 60)
        
        interaction_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        interaction = {
            'user_id': user_id,
            'article_id': article_id,
            'event': event,
            'ts': interaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'dwell_time_s': dwell_time
        }
        interactions_data.append(interaction)
    
    interactions_df = pd.DataFrame(interactions_data)
    
    # Remove duplicate user-article-event combinations (keep most recent)
    interactions_df = interactions_df.sort_values('ts').drop_duplicates(
        subset=['user_id', 'article_id', 'event'], keep='last'
    )
    
    logger.info(f"Generated {len(articles_df)} articles, {len(users_df)} users, {len(interactions_df)} interactions")
    
    return articles_df, users_df, interactions_df


def load_kaggle_data(articles_path: str, interactions_path: str = None):
    """Load data from Kaggle CSV files."""
    logger.info(f"Loading articles from {articles_path}")
    
    # Load articles
    articles_df = pd.read_csv(articles_path)
    
    # Standardize column names if necessary
    column_mapping = {
        'id': 'article_id',
        'headline': 'title', 
        'short_description': 'summary',
        'link': 'url',
        'category': 'topics',
        'date': 'published_at',
        'authors': 'authors'
    }
    
    articles_df = articles_df.rename(columns={k: v for k, v in column_mapping.items() if k in articles_df.columns})
    
    # Ensure required columns exist
    required_columns = ['article_id', 'title', 'source', 'published_at']
    for col in required_columns:
        if col not in articles_df.columns:
            if col == 'source':
                articles_df['source'] = 'unknown'
            elif col == 'article_id':
                articles_df['article_id'] = range(1, len(articles_df) + 1)
            else:
                articles_df[col] = ''
    
    # Fill missing values
    articles_df['summary'] = articles_df['summary'].fillna('')
    articles_df['content'] = articles_df.get('content', articles_df['summary'])
    articles_df['authors'] = articles_df['authors'].fillna('Unknown')
    articles_df['topics'] = articles_df['topics'].fillna('general')
    articles_df['url'] = articles_df['url'].fillna('')
    
    logger.info(f"Loaded {len(articles_df)} articles")
    
    # Load interactions if provided
    interactions_df = None
    users_df = None
    
    if interactions_path and Path(interactions_path).exists():
        logger.info(f"Loading interactions from {interactions_path}")
        interactions_df = pd.read_csv(interactions_path)
        
        # Generate users from interactions
        unique_users = interactions_df['user_id'].unique()
        users_data = []
        
        for user_id in unique_users:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            # Infer preferred topics from interaction history
            if 'topics' in articles_df.columns:
                user_articles = articles_df[articles_df['article_id'].isin(user_interactions['article_id'])]
                preferred_topics = user_articles['topics'].value_counts().head(3).index.tolist()
            else:
                preferred_topics = np.random.choice(TOPICS, size=2, replace=False).tolist()
            
            users_data.append({
                'user_id': user_id,
                'signup_ts': user_interactions['ts'].min() if 'ts' in user_interactions.columns else datetime.now(),
                'declared_topics': ','.join(preferred_topics)
            })
        
        users_df = pd.DataFrame(users_data)
        logger.info(f"Generated {len(users_df)} users from interactions")
    
    return articles_df, users_df, interactions_df


def ingest_data(articles_df: pd.DataFrame, 
               users_df: pd.DataFrame = None, 
               interactions_df: pd.DataFrame = None,
               db_path: str = None):
    """Ingest data into the database."""
    logger.info("Starting data ingestion...")
    
    # Initialize database
    db = NewsDatabase() if db_path is None else NewsDatabase(Path(db_path))
    
    # Insert articles
    logger.info("Inserting articles...")
    try:
        db.insert_articles(articles_df)
        logger.info(f"Successfully inserted {len(articles_df)} articles")
    except Exception as e:
        logger.error(f"Error inserting articles: {e}")
        return False
    
    # Insert users if provided
    if users_df is not None:
        logger.info("Inserting users...")
        try:
            db.insert_users(users_df)
            logger.info(f"Successfully inserted {len(users_df)} users")
        except Exception as e:
            logger.error(f"Error inserting users: {e}")
            return False
    
    # Insert interactions if provided
    if interactions_df is not None:
        logger.info("Inserting interactions...")
        try:
            db.insert_interactions(interactions_df)
            logger.info(f"Successfully inserted {len(interactions_df)} interactions")
        except Exception as e:
            logger.error(f"Error inserting interactions: {e}")
            return False
    
    # Execute SQL queries for verification
    logger.info("Running verification queries...")
    try:
        results = db.execute_sql_queries()
        for query_name, result_df in results.items():
            logger.info(f"{query_name}: {len(result_df)} rows")
            if len(result_df) > 0:
                print(f"\nSample results for {query_name}:")
                print(result_df.head().to_string())
    except Exception as e:
        logger.warning(f"Error running verification queries: {e}")
    
    logger.info("Data ingestion completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Ingest news data into database')
    parser.add_argument('--articles', type=str, help='Path to articles CSV file')
    parser.add_argument('--interactions', type=str, help='Path to interactions CSV file')
    parser.add_argument('--users', type=str, help='Path to users CSV file') 
    parser.add_argument('--generate-sample', action='store_true', 
                       help='Generate sample data instead of loading from files')
    parser.add_argument('--n-articles', type=int, default=1000,
                       help='Number of sample articles to generate')
    parser.add_argument('--n-users', type=int, default=200,
                       help='Number of sample users to generate')
    parser.add_argument('--n-interactions', type=int, default=5000,
                       help='Number of sample interactions to generate')
    parser.add_argument('--db-path', type=str, help='Path to database file')
    
    args = parser.parse_args()
    
    try:
        if args.generate_sample:
            # Generate sample data
            articles_df, users_df, interactions_df = generate_sample_data(
                args.n_articles, args.n_users, args.n_interactions
            )
        else:
            # Load from files
            if not args.articles:
                # Try to find articles file in raw data directory
                articles_path = RAW_DATA_DIR / "articles.csv"
                if not articles_path.exists():
                    logger.error("No articles file specified and articles.csv not found in raw data directory")
                    logger.info("Either provide --articles path or use --generate-sample")
                    return
                args.articles = str(articles_path)
            
            articles_df, users_df, interactions_df = load_kaggle_data(
                args.articles, args.interactions
            )
        
        # Ingest data
        success = ingest_data(articles_df, users_df, interactions_df, args.db_path)
        
        if success:
            logger.info("Data ingestion completed successfully!")
        else:
            logger.error("Data ingestion failed!")
            return
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
