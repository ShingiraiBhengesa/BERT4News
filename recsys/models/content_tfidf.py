"""Content-based recommendation using TF-IDF vectors."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib

from ..config import TFIDF_CONFIG, ARTIFACTS_DIR, SOURCE_QUALITY_SCORES
from ..utils.text import TextProcessor, preprocess_articles_batch
from ..db import NewsDatabase

logger = logging.getLogger(__name__)


class TFIDFContentModel:
    """Content-based recommendation using TF-IDF and cosine similarity."""
    
    def __init__(self, config: dict = None):
        self.config = config or TFIDF_CONFIG
        self.vectorizer = TfidfVectorizer(**self.config)
        self.text_processor = TextProcessor()
        self.tfidf_matrix = None
        self.article_ids = None
        self.article_features = None
        self.is_fitted = False
        
        # Paths for saving/loading models
        self.model_dir = Path(ARTIFACTS_DIR) / "tfidf_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, articles_df: pd.DataFrame, text_columns: List[str] = None) -> 'TFIDFContentModel':
        """Fit the TF-IDF model on articles data."""
        if text_columns is None:
            text_columns = ['title', 'summary', 'content']
        
        logger.info(f"Fitting TF-IDF model on {len(articles_df)} articles")
        
        # Preprocess articles
        processed_df = preprocess_articles_batch(articles_df, text_columns)
        
        # Extract processed text
        documents = processed_df['processed_text'].fillna('').tolist()
        
        # Fit TF-IDF vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.article_ids = processed_df['article_id'].tolist()
        
        # Store additional article features for enhanced recommendations
        self.article_features = processed_df[['article_id', 'source', 'published_at', 'topics']].copy()
        
        # Add source quality scores
        self.article_features['source_quality'] = self.article_features['source'].map(
            lambda x: SOURCE_QUALITY_SCORES.get(x.lower(), SOURCE_QUALITY_SCORES['default'])
        )
        
        # Add recency scores (more recent articles get higher scores)
        self.article_features['published_at'] = pd.to_datetime(self.article_features['published_at'])
        max_date = self.article_features['published_at'].max()
        self.article_features['recency_score'] = (
            1 - (max_date - self.article_features['published_at']).dt.days / 30
        ).clip(0, 1)
        
        self.is_fitted = True
        logger.info("TF-IDF model fitted successfully")
        return self
    
    def get_similar_articles(self, 
                           article_id: int, 
                           n_recommendations: int = 10,
                           include_scores: bool = True) -> List[Dict]:
        """Get articles similar to a given article."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if article_id not in self.article_ids:
            logger.warning(f"Article {article_id} not found in training data")
            return []
        
        # Get article index
        article_idx = self.article_ids.index(article_id)
        
        # Calculate cosine similarity with all articles
        article_vector = self.tfidf_matrix[article_idx]
        similarities = cosine_similarity(article_vector, self.tfidf_matrix).flatten()
        
        # Get top similar articles (excluding the input article)
        similar_indices = similarities.argsort()[::-1]
        similar_indices = [idx for idx in similar_indices if idx != article_idx]
        
        recommendations = []
        for idx in similar_indices[:n_recommendations]:
            similar_article_id = self.article_ids[idx]
            similarity_score = similarities[idx]
            
            # Get article features
            article_info = self.article_features[
                self.article_features['article_id'] == similar_article_id
            ].iloc[0]
            
            # Apply source quality and recency boosts
            boosted_score = (similarity_score * 
                           article_info['source_quality'] * 
                           (1 + 0.1 * article_info['recency_score']))
            
            recommendation = {
                'article_id': similar_article_id,
                'content_score': similarity_score,
                'boosted_score': boosted_score,
                'source': article_info['source'],
                'topics': article_info['topics'],
                'recency_score': article_info['recency_score']
            }
            
            if include_scores:
                recommendation['explanation'] = self._generate_explanation(
                    article_idx, idx, similarity_score
                )
            
            recommendations.append(recommendation)
        
        # Sort by boosted score
        recommendations.sort(key=lambda x: x['boosted_score'], reverse=True)
        return recommendations
    
    def get_recommendations_for_user_profile(self,
                                           user_topics: List[str],
                                           user_keywords: List[str] = None,
                                           n_recommendations: int = 100,
                                           days_back: int = 30) -> List[Dict]:
        """Get content-based recommendations for user profile."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Create user profile vector
        user_profile = self._create_user_profile_vector(user_topics, user_keywords)
        
        # Calculate similarities with all articles
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        
        # Filter by recency
        recent_mask = self.article_features['recency_score'] > 0.1  # Articles from last ~30 days
        
        recommendations = []
        for idx, similarity_score in enumerate(similarities):
            if not recent_mask.iloc[idx]:
                continue
                
            article_id = self.article_ids[idx]
            article_info = self.article_features[
                self.article_features['article_id'] == article_id
            ].iloc[0]
            
            # Topic matching bonus
            article_topics = article_info['topics'].split(',') if article_info['topics'] else []
            topic_match_score = len(set(user_topics) & set(article_topics)) / max(len(user_topics), 1)
            
            # Apply boosts
            boosted_score = (similarity_score * 
                           article_info['source_quality'] * 
                           (1 + 0.2 * topic_match_score) *
                           (1 + 0.1 * article_info['recency_score']))
            
            recommendations.append({
                'article_id': article_id,
                'content_score': similarity_score,
                'topic_match_score': topic_match_score,
                'boosted_score': boosted_score,
                'source': article_info['source'],
                'topics': article_info['topics']
            })
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x['boosted_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_topic_based_recommendations(self, 
                                      topics: List[str], 
                                      n_recommendations: int = 50) -> List[Dict]:
        """Get recommendations based purely on topic matching."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = []
        
        for idx, article_id in enumerate(self.article_ids):
            article_info = self.article_features[
                self.article_features['article_id'] == article_id
            ].iloc[0]
            
            if not article_info['topics']:
                continue
                
            article_topics = [t.strip().lower() for t in article_info['topics'].split(',')]
            user_topics_lower = [t.lower() for t in topics]
            
            # Calculate topic match score
            matches = set(article_topics) & set(user_topics_lower)
            if not matches:
                continue
                
            topic_score = len(matches) / len(set(article_topics) | set(user_topics_lower))
            
            # Apply source quality and recency boosts
            final_score = (topic_score * 
                          article_info['source_quality'] * 
                          (1 + 0.2 * article_info['recency_score']))
            
            recommendations.append({
                'article_id': article_id,
                'topic_score': topic_score,
                'final_score': final_score,
                'matched_topics': list(matches),
                'source': article_info['source']
            })
        
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _create_user_profile_vector(self, 
                                  topics: List[str], 
                                  keywords: List[str] = None) -> csr_matrix:
        """Create a TF-IDF vector representing user's profile."""
        # Combine topics and keywords into a profile text
        profile_parts = topics.copy()
        if keywords:
            profile_parts.extend(keywords)
        
        profile_text = ' '.join(profile_parts)
        
        # Preprocess the profile text
        processed_profile = self.text_processor.preprocess_for_tfidf(profile_text)
        
        # Transform using fitted vectorizer
        profile_vector = self.vectorizer.transform([processed_profile])
        
        return profile_vector
    
    def _generate_explanation(self, 
                            source_idx: int, 
                            target_idx: int, 
                            similarity_score: float) -> Dict:
        """Generate explanation for why an article was recommended."""
        # Get TF-IDF vectors for both articles
        source_vector = self.tfidf_matrix[source_idx].toarray().flatten()
        target_vector = self.tfidf_matrix[target_idx].toarray().flatten()
        
        # Find top contributing terms
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate element-wise product to find matching important terms
        term_contributions = source_vector * target_vector
        
        # Get top contributing terms
        top_indices = term_contributions.argsort()[::-1][:5]
        top_terms = [feature_names[i] for i in top_indices if term_contributions[i] > 0]
        
        return {
            'similarity_score': float(similarity_score),
            'top_matching_terms': top_terms,
            'explanation': f"Similar content with shared terms: {', '.join(top_terms[:3])}"
        }
    
    def save_model(self, model_name: str = "tfidf_content_model") -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_path = self.model_dir / f"{model_name}.pkl"
        
        model_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'article_ids': self.article_ids,
            'article_features': self.article_features,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = "tfidf_content_model") -> 'TFIDFContentModel':
        """Load a fitted model from disk."""
        model_path = self.model_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.article_ids = model_data['article_ids']
        self.article_features = model_data['article_features']
        self.config = model_data['config']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {model_path}")
        return self
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'n_articles': len(self.article_ids),
            'n_features': self.tfidf_matrix.shape[1],
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'sparsity': 1 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])),
            'config': self.config
        }
    
    def compute_diversity_score(self, article_ids: List[int]) -> float:
        """Compute diversity score for a list of articles (lower = more diverse)."""
        if len(article_ids) < 2:
            return 0.0
        
        # Get indices for the articles
        indices = []
        for article_id in article_ids:
            if article_id in self.article_ids:
                indices.append(self.article_ids.index(article_id))
        
        if len(indices) < 2:
            return 0.0
        
        # Get TF-IDF vectors for these articles
        vectors = self.tfidf_matrix[indices]
        
        # Compute pairwise similarities
        similarities = cosine_similarity(vectors)
        
        # Calculate average pairwise similarity (excluding diagonal)
        n = len(indices)
        total_similarity = similarities.sum() - np.trace(similarities)  # Exclude diagonal
        avg_similarity = total_similarity / (n * (n - 1))
        
        return float(avg_similarity)
