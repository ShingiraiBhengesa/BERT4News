"""Hybrid recommendation system combining collaborative and content-based filtering."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import joblib
from collections import defaultdict

from .content_tfidf import TFIDFContentModel
from .cf_surprise import CollaborativeFilteringModel
from ..config import RECOMMENDATION_CONFIG, ARTIFACTS_DIR
from ..db import NewsDatabase

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid recommendation system combining CF and content-based approaches."""
    
    def __init__(self, 
                 cf_weight: float = 0.6,
                 content_weight: float = 0.4,
                 config: dict = None):
        """Initialize hybrid recommender.
        
        Args:
            cf_weight: Weight for collaborative filtering scores
            content_weight: Weight for content-based scores
            config: Configuration dictionary
        """
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.config = config or RECOMMENDATION_CONFIG
        
        # Component models
        self.cf_model = CollaborativeFilteringModel()
        self.content_model = TFIDFContentModel()
        
        # Fitted state
        self.is_fitted = False
        
        # Database connection
        self.db = NewsDatabase()
        
        # Caching for performance
        self._user_cache = {}
        self._article_cache = {}
        
        # Model artifacts directory
        self.model_dir = Path(ARTIFACTS_DIR) / "hybrid_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, 
            articles_df: pd.DataFrame,
            interactions_df: pd.DataFrame,
            validate_weights: bool = True) -> 'HybridRecommender':
        """Fit both collaborative and content-based models."""
        logger.info("Fitting hybrid recommendation system")
        
        # Fit content model
        logger.info("Fitting content-based model...")
        self.content_model.fit(articles_df)
        
        # Fit collaborative filtering model
        logger.info("Fitting collaborative filtering model...")
        self.cf_model.fit(interactions_df)
        
        # Optionally validate and optimize weights
        if validate_weights:
            self._optimize_weights(articles_df, interactions_df)
        
        self.is_fitted = True
        logger.info(f"Hybrid model fitted with weights: CF={self.cf_weight}, Content={self.content_weight}")
        return self
    
    def _optimize_weights(self, articles_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Optimize CF and content weights using validation data."""
        logger.info("Optimizing hybrid model weights...")
        
        # Create validation set (last 20% of interactions chronologically)
        interactions_sorted = interactions_df.sort_values('ts')
        split_idx = int(len(interactions_sorted) * 0.8)
        val_interactions = interactions_sorted.iloc[split_idx:].copy()
        
        # Test different weight combinations
        weight_combinations = [
            (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)
        ]
        
        best_score = 0
        best_weights = (0.6, 0.4)
        
        for cf_w, content_w in weight_combinations:
            temp_cf_weight, temp_content_weight = self.cf_weight, self.content_weight
            self.cf_weight, self.content_weight = cf_w, content_w
            
            # Evaluate on validation set
            score = self._evaluate_on_validation(val_interactions)
            
            if score > best_score:
                best_score = score
                best_weights = (cf_w, content_w)
            
            # Restore original weights
            self.cf_weight, self.content_weight = temp_cf_weight, temp_content_weight
        
        # Set best weights
        self.cf_weight, self.content_weight = best_weights
        logger.info(f"Optimized weights: CF={self.cf_weight}, Content={self.content_weight}, Score={best_score:.4f}")
    
    def _evaluate_on_validation(self, val_interactions: pd.DataFrame, k: int = 10) -> float:
        """Evaluate hybrid model on validation set using Precision@K."""
        if len(val_interactions) == 0:
            return 0.0
        
        users_to_eval = val_interactions['user_id'].unique()
        if len(users_to_eval) > 100:  # Limit for performance
            users_to_eval = np.random.choice(users_to_eval, 100, replace=False)
        
        precisions = []
        
        for user_id in users_to_eval:
            user_val = val_interactions[val_interactions['user_id'] == user_id]
            true_articles = set(user_val['article_id'].tolist())
            
            if len(true_articles) == 0:
                continue
                
            try:
                recommendations = self.recommend(user_id, n_recommendations=k)
                predicted_articles = set([rec['article_id'] for rec in recommendations])
                
                if len(predicted_articles) > 0:
                    precision = len(true_articles & predicted_articles) / len(predicted_articles)
                    precisions.append(precision)
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        return np.mean(precisions) if precisions else 0.0
    
    def recommend(self, 
                  user_id: int,
                  n_recommendations: int = 10,
                  diversity_weight: float = 0.1,
                  include_explanations: bool = True,
                  exclude_seen: bool = True) -> List[Dict]:
        """Generate hybrid recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        cf_recommendations = self._get_cf_candidates(user_id, exclude_seen)
        content_recommendations = self._get_content_candidates(user_id)
        
        # Combine recommendations
        combined_recommendations = self._combine_recommendations(
            cf_recommendations, content_recommendations, user_id
        )
        
        # Apply diversity filter if requested
        if diversity_weight > 0:
            combined_recommendations = self._apply_diversity_filter(
                combined_recommendations, diversity_weight
            )
        
        # Add explanations
        if include_explanations:
            combined_recommendations = self._add_explanations(
                combined_recommendations, user_id
            )
        
        # Apply final business rules
        final_recommendations = self._apply_business_rules(
            combined_recommendations, user_id
        )
        
        return final_recommendations[:n_recommendations]
    
    def _get_cf_candidates(self, user_id: int, exclude_seen: bool) -> List[Dict]:
        """Get collaborative filtering candidates."""
        n_candidates = self.config['n_candidates_cf']
        
        try:
            cf_recs = self.cf_model.get_user_recommendations(
                user_id, n_candidates, exclude_seen
            )
            return cf_recs
        except Exception as e:
            logger.warning(f"CF recommendations failed for user {user_id}: {e}")
            return []
    
    def _get_content_candidates(self, user_id: int) -> List[Dict]:
        """Get content-based candidates."""
        n_candidates = self.config['n_candidates_content']
        
        try:
            # Get user's topic preferences
            user_topics = self.db.get_user_topic_affinity(user_id)
            
            if not user_topics:
                # Fallback to popular topics
                user_topics = ['technology', 'politics', 'business']
            else:
                user_topics = list(user_topics.keys())
            
            content_recs = self.content_model.get_recommendations_for_user_profile(
                user_topics, n_recommendations=n_candidates
            )
            return content_recs
        except Exception as e:
            logger.warning(f"Content recommendations failed for user {user_id}: {e}")
            return []
    
    def _combine_recommendations(self, 
                               cf_recs: List[Dict], 
                               content_recs: List[Dict],
                               user_id: int) -> List[Dict]:
        """Combine CF and content recommendations with weighted scores."""
        # Create article score mappings
        cf_scores = {rec['article_id']: rec.get('cf_score', 0) for rec in cf_recs}
        content_scores = {rec['article_id']: rec.get('boosted_score', rec.get('content_score', 0)) 
                         for rec in content_recs}
        
        # Get all unique articles
        all_articles = set(cf_scores.keys()) | set(content_scores.keys())
        
        combined_recs = []
        
        for article_id in all_articles:
            cf_score = cf_scores.get(article_id, 0)
            content_score = content_scores.get(article_id, 0)
            
            # Normalize scores to 0-1 range
            cf_norm = self._normalize_score(cf_score, 1, 5) if cf_score > 0 else 0
            content_norm = self._normalize_score(content_score, 0, 1) if content_score > 0 else 0
            
            # Calculate hybrid score
            hybrid_score = (self.cf_weight * cf_norm + 
                           self.content_weight * content_norm)
            
            # Get article metadata
            article_info = self._get_article_info(article_id)
            
            recommendation = {
                'article_id': article_id,
                'hybrid_score': hybrid_score,
                'cf_score': cf_score,
                'content_score': content_score,
                'cf_norm': cf_norm,
                'content_norm': content_norm,
                'source': article_info.get('source', 'unknown'),
                'topics': article_info.get('topics', ''),
                'published_at': article_info.get('published_at', ''),
                'title': article_info.get('title', ''),
                'summary': article_info.get('summary', '')
            }
            
            combined_recs.append(recommendation)
        
        # Sort by hybrid score
        combined_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return combined_recs
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return max(0, min(1, (score - min_val) / (max_val - min_val)))
    
    def _get_article_info(self, article_id: int) -> Dict:
        """Get article information from database or cache."""
        if article_id in self._article_cache:
            return self._article_cache[article_id]
        
        # Query database
        query = "SELECT * FROM articles WHERE article_id = ?"
        with self.db.get_connection() as conn:
            result = conn.execute(query, [article_id]).fetchone()
        
        if result:
            article_info = dict(result)
            self._article_cache[article_id] = article_info
            return article_info
        
        return {}
    
    def _apply_diversity_filter(self, 
                              recommendations: List[Dict], 
                              diversity_weight: float) -> List[Dict]:
        """Apply diversity filtering to reduce similarity in recommendations."""
        if len(recommendations) <= 1:
            return recommendations
        
        # Extract article IDs for diversity computation
        article_ids = [rec['article_id'] for rec in recommendations]
        
        # Compute diversity scores using content model
        try:
            diversity_penalty = self.content_model.compute_diversity_score(article_ids)
        except:
            diversity_penalty = 0.0
        
        # Apply MMR-style re-ranking
        diversified_recs = []
        remaining_recs = recommendations.copy()
        
        # Always include the top recommendation
        if remaining_recs:
            diversified_recs.append(remaining_recs.pop(0))
        
        # Iteratively select diverse recommendations
        while remaining_recs and len(diversified_recs) < len(recommendations):
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining_recs):
                # Calculate diversity with already selected items
                selected_ids = [rec['article_id'] for rec in diversified_recs]
                candidate_diversity = self._calculate_diversity_with_selected(
                    candidate['article_id'], selected_ids
                )
                
                # MMR formula: balance relevance and diversity
                mmr_score = (
                    (1 - diversity_weight) * candidate['hybrid_score'] +
                    diversity_weight * candidate_diversity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            diversified_recs.append(remaining_recs.pop(best_idx))
        
        return diversified_recs
    
    def _calculate_diversity_with_selected(self, 
                                         candidate_id: int, 
                                         selected_ids: List[int]) -> float:
        """Calculate how diverse a candidate is compared to selected items."""
        if not selected_ids:
            return 1.0
        
        try:
            # Use content model to compute similarity
            similarities = []
            for selected_id in selected_ids:
                similar_articles = self.content_model.get_similar_articles(
                    selected_id, n_recommendations=1000, include_scores=False
                )
                for sim_article in similar_articles:
                    if sim_article['article_id'] == candidate_id:
                        similarities.append(sim_article['content_score'])
                        break
                else:
                    similarities.append(0.0)  # No similarity found
            
            # Diversity is 1 - average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            return 1.0 - avg_similarity
        except:
            return 0.5  # Default diversity score
    
    def _add_explanations(self, 
                         recommendations: List[Dict], 
                         user_id: int) -> List[Dict]:
        """Add explanations for why articles were recommended."""
        for rec in recommendations:
            explanation_parts = []
            
            # CF explanation
            if rec['cf_score'] > 0:
                explanation_parts.append(
                    f"Users with similar preferences liked this (CF score: {rec['cf_norm']:.2f})"
                )
            
            # Content explanation
            if rec['content_score'] > 0:
                # Get user's topic preferences
                user_topics = self.db.get_user_topic_affinity(user_id)
                article_topics = rec['topics'].split(',') if rec['topics'] else []
                
                matching_topics = []
                for topic in article_topics:
                    topic_clean = topic.strip().lower()
                    if topic_clean in [t.lower() for t in user_topics.keys()]:
                        matching_topics.append(topic.strip())
                
                if matching_topics:
                    explanation_parts.append(
                        f"Matches your interests: {', '.join(matching_topics[:3])}"
                    )
                else:
                    explanation_parts.append(
                        f"Similar content to articles you've read (Content score: {rec['content_norm']:.2f})"
                    )
            
            # Source explanation
            if rec['source']:
                source_quality = self._get_source_quality_score(rec['source'])
                if source_quality > 1.0:
                    explanation_parts.append(f"From trusted source: {rec['source']}")
            
            rec['explanation'] = " | ".join(explanation_parts) if explanation_parts else "Recommended for you"
            
        return recommendations
    
    def _get_source_quality_score(self, source: str) -> float:
        """Get quality score for a news source."""
        from ..config import SOURCE_QUALITY_SCORES
        return SOURCE_QUALITY_SCORES.get(source.lower(), SOURCE_QUALITY_SCORES['default'])
    
    def _apply_business_rules(self, 
                            recommendations: List[Dict], 
                            user_id: int) -> List[Dict]:
        """Apply business rules and final filtering."""
        filtered_recs = []
        
        # Track sources to ensure diversity
        source_counts = defaultdict(int)
        max_per_source = 3
        
        # Track topics to ensure diversity
        topic_counts = defaultdict(int)
        max_per_topic = 4
        
        for rec in recommendations:
            # Source diversity rule
            source = rec.get('source', 'unknown')
            if source_counts[source] >= max_per_source:
                continue
            
            # Topic diversity rule  
            topics = rec.get('topics', '').split(',')
            topic_violation = False
            for topic in topics:
                topic_clean = topic.strip().lower()
                if topic_clean and topic_counts[topic_clean] >= max_per_topic:
                    topic_violation = True
                    break
            
            if topic_violation:
                continue
            
            # Apply recency filter (prefer recent articles)
            try:
                published_at = pd.to_datetime(rec.get('published_at'))
                days_old = (pd.Timestamp.now() - published_at).days
                
                if days_old > 30:  # Skip articles older than 30 days
                    rec['hybrid_score'] *= 0.8  # Reduce score for old articles
                elif days_old > 7:  # Reduce score for week-old articles
                    rec['hybrid_score'] *= 0.9
            except:
                pass
            
            # Update counters
            source_counts[source] += 1
            for topic in topics:
                topic_clean = topic.strip().lower()
                if topic_clean:
                    topic_counts[topic_clean] += 1
            
            filtered_recs.append(rec)
        
        # Re-sort after applying business rules
        filtered_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return filtered_recs
    
    def recommend_cold_start(self, 
                           user_topics: List[str],
                           n_recommendations: int = 10) -> List[Dict]:
        """Generate recommendations for new users based on declared topics."""
        logger.info(f"Generating cold start recommendations for topics: {user_topics}")
        
        # Get content-based recommendations
        content_recs = self.content_model.get_topic_based_recommendations(
            user_topics, n_recommendations * 2
        )
        
        # Get popular articles as fallback
        popular_articles = self.db.get_popular_articles(days=7, limit=50)
        
        recommendations = []
        
        # Combine content and popular recommendations
        for rec in content_recs:
            article_info = self._get_article_info(rec['article_id'])
            
            recommendation = {
                'article_id': rec['article_id'],
                'hybrid_score': rec['final_score'] * 0.8,  # Lower confidence for cold start
                'cf_score': 0,
                'content_score': rec['topic_score'],
                'method': 'cold_start_content',
                'matched_topics': rec.get('matched_topics', []),
                'source': rec.get('source', article_info.get('source', '')),
                'topics': article_info.get('topics', ''),
                'title': article_info.get('title', ''),
                'explanation': f"Popular in topics you selected: {', '.join(rec.get('matched_topics', [])[:2])}"
            }
            recommendations.append(recommendation)
        
        # Add popular articles if needed
        existing_ids = set(rec['article_id'] for rec in recommendations)
        for _, article in popular_articles.iterrows():
            if len(recommendations) >= n_recommendations:
                break
                
            if article['article_id'] not in existing_ids:
                recommendation = {
                    'article_id': article['article_id'],
                    'hybrid_score': 0.5,  # Medium confidence
                    'cf_score': 0,
                    'content_score': 0,
                    'method': 'cold_start_popular',
                    'source': article['source'],
                    'topics': article.get('topics', ''),
                    'title': article['title'],
                    'explanation': "Popular recent article"
                }
                recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def save_model(self, model_name: str = "hybrid_recommender") -> None:
        """Save the hybrid model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save individual models
        self.cf_model.save_model("cf_model_hybrid")
        self.content_model.save_model("content_model_hybrid")
        
        # Save hybrid-specific configuration
        model_path = self.model_dir / f"{model_name}.pkl"
        
        hybrid_data = {
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(hybrid_data, model_path)
        logger.info(f"Hybrid model saved to {model_path}")
    
    def load_model(self, model_name: str = "hybrid_recommender") -> 'HybridRecommender':
        """Load the hybrid model from disk."""
        model_path = self.model_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load individual models
        self.cf_model.load_model("cf_model_hybrid")
        self.content_model.load_model("content_model_hybrid")
        
        # Load hybrid configuration
        hybrid_data = joblib.load(model_path)
        
        self.cf_weight = hybrid_data['cf_weight']
        self.content_weight = hybrid_data['content_weight']
        self.config = hybrid_data['config']
        self.is_fitted = hybrid_data['is_fitted']
        
        logger.info(f"Hybrid model loaded from {model_path}")
        return self
    
    def get_model_info(self) -> Dict:
        """Get information about the hybrid model."""
        if not self.is_fitted:
            return {'fitted': False}
        
        info = {
            'fitted': True,
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'config': self.config
        }
        
        # Add component model info
        if self.cf_model.is_fitted:
            info['cf_model'] = self.cf_model.get_model_info()
        
        if self.content_model.is_fitted:
            info['content_model'] = self.content_model.get_model_info()
        
        return info
