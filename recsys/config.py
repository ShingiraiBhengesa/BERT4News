"""Configuration settings for the recommendation system."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# Database configuration
DATABASE_PATH = DATA_DIR / "news_recommendations.db"

# Model parameters
TFIDF_CONFIG = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'max_df': 0.8,
    'min_df': 2,
    'stop_words': 'english'
}

CF_CONFIG = {
    'algorithm': 'SVD',
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02
}

EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'max_seq_length': 512,
    'batch_size': 32
}

FAISS_CONFIG = {
    'index_type': 'IndexFlatIP',  # Inner Product for cosine similarity
    'n_candidates': 100
}

RERANKER_CONFIG = {
    'model_type': 'lightgbm',
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Recommendation parameters
RECOMMENDATION_CONFIG = {
    'n_candidates_cf': 100,
    'n_candidates_content': 100,
    'final_recommendations': 10,
    'diversity_weight': 0.1,
    'recency_decay_days': 30,
    'min_article_interactions': 5
}

# Evaluation parameters
EVAL_CONFIG = {
    'test_size': 0.2,
    'temporal_split_days': 14,
    'k_values': [5, 10, 20],
    'min_interactions_per_user': 10
}

# Flask configuration
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'DEBUG': os.environ.get('FLASK_DEBUG', 'True').lower() == 'true',
    'CACHE_TYPE': 'redis' if os.environ.get('REDIS_URL') else 'simple',
    'CACHE_DEFAULT_TIMEOUT': int(os.environ.get('CACHE_EXPIRY_HOURS', 24)) * 3600
}

# Topics for categorization
TOPICS = [
    'technology', 'politics', 'business', 'sports', 'entertainment',
    'health', 'science', 'world', 'finance', 'lifestyle'
]

# Source quality scores (can be learned from data)
SOURCE_QUALITY_SCORES = {
    'reuters': 1.2,
    'bbc': 1.2,
    'cnn': 1.1,
    'nytimes': 1.2,
    'guardian': 1.1,
    'default': 1.0
}
