"""News recommendation system package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import *
from .db import NewsDatabase

__all__ = [
    'NewsDatabase',
    'TFIDF_CONFIG',
    'CF_CONFIG',
    'EMBEDDING_CONFIG',
    'RECOMMENDATION_CONFIG',
    'TOPICS'
]
