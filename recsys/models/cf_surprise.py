"""Collaborative filtering using Surprise SVD algorithm."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import pickle
import logging
from pathlib import Path
import joblib
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.prediction_algorithms import AlgoBase
from collections import defaultdict

from ..config import CF_CONFIG, ARTIFACTS_DIR, RECOMMENDATION_CONFIG
from ..db import NewsDatabase

logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """Collaborative filtering using matrix factorization (SVD)."""
    
    def __init__(self, config: dict = None):
        self.config = config or CF_CONFIG
        self.model = SVD(
            n_factors=self.config['n_factors'],
            n_epochs=self.config['n_epochs'],
            lr_all=self.config['lr_all'],
            reg_all=self.config['reg_all']
        )
        self.trainset = None
        self.user_item_matrix = None
        self.user_ids = None
        self.article_ids = None
        self.is_fitted = False
        
        # For cold start handling
        self.user_profiles = {}
        self.popular_articles = []
        
        # Model artifacts directory
        self.model_dir = Path(ARTIFACTS_DIR) / "cf_model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def fit(self, interactions_df: pd.DataFrame, 
            min_interactions_per_user: int = 5,
            min_interactions_per_article: int = 3) -> 'CollaborativeFilteringModel':
        """Fit the collaborative filtering model."""
        logger.info(f"Fitting CF model on {len(interactions_df)} interactions")
        
        # Filter users and articles with minimum interactions
        user_counts = interactions_df['user_id'].value_counts()
        article_counts = interactions_df['article_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions_per_user].index
        valid_articles = article_counts[article_counts >= min_interactions_per_article].index
        
        filtered_df = interactions_df[
            (interactions_df['user_id'].isin(valid_users)) &
            (interactions_df['article_id'].isin(valid_articles))
        ].copy()
        
        logger.info(f"After filtering: {len(filtered_df)} interactions, "
                   f"{filtered_df['user_id'].nunique()} users, "
                   f"{filtered_df['article_id'].nunique()} articles")
        
        # Create implicit ratings based on interaction types
        filtered_df['rating'] = filtered_df['event'].map({
            'like': 5.0,
            'read': 4.0,
            'click': 3.0,
            'share': 5.0,
            'dislike': 1.0
        }).fillna(3.0)
        
        # Aggregate multiple interactions per user-article pair
        user_item_ratings = filtered_df.groupby(['user_id', 'article_id'])['rating'].mean().reset_index()
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(user_item_ratings, reader)
        
        # Build full trainset
        self.trainset = data.build_full_trainset()
        
        # Fit the model
        self.model.fit(self.trainset)
        
        # Store mappings
        self.user_ids = list(self.trainset.all_users())
        self.article_ids = list(self.trainset.all_items())
        
        # Create user-item matrix for analysis
        self._create_user_item_matrix(user_item_ratings)
        
        # Prepare cold start data
        self._prepare_cold_start_data(filtered_df)
        
        self.is_fitted = True
        logger.info("CF model fitted successfully")
        return self
    
    def _create_user_item_matrix(self, user_item_ratings: pd.DataFrame):
        """Create user-item matrix for analysis."""
        self.user_item_matrix = user_item_ratings.pivot(
            index='user_id', 
            columns='article_id', 
            values='rating'
        ).fillna(0)
        
        logger.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
    
    def _prepare_cold_start_data(self, interactions_df: pd.DataFrame):
        """Prepare data for handling cold start problems."""
        # Get popular articles (for new users)
        popularity_scores = interactions_df.groupby('article_id').agg({
            'user_id': 'nunique',  # Number of unique users
            'rating': 'mean'       # Average rating
        }).rename(columns={'user_id': 'n_users', 'rating': 'avg_rating'})
        
        # Calculate popularity score (combining user count and rating)
        popularity_scores['popularity_score'] = (
            0.7 * (popularity_scores['n_users'] / popularity_scores['n_users'].max()) +
            0.3 * (popularity_scores['avg_rating'] / 5.0)
        )
        
        self.popular_articles = popularity_scores.sort_values(
            'popularity_score', ascending=False
        ).index.tolist()
        
        # Create user profiles (for new articles)
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            self.user_profiles[user_id] = {
                'avg_rating': user_interactions['rating'].mean(),
                'interaction_count': len(user_interactions),
                'preferred_articles': user_interactions.nlargest(10, 'rating')['article_id'].tolist()
            }
    
    def predict(self, user_id: int, article_id: int) -> float:
        """Predict rating for a user-article pair."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Check if user and article are in training data
        if user_id not in self.trainset.all_users() or article_id not in self.trainset.all_items():
            return self._handle_cold_start_prediction(user_id, article_id)
        
        prediction = self.model.predict(user_id, article_id)
        return prediction.est
    
    def _handle_cold_start_prediction(self, user_id: int, article_id: int) -> float:
        """Handle predictions for new users or articles."""
        # New user, known article
        if user_id not in self.trainset.all_users() and article_id in self.trainset.all_items():
            # Return average rating for this article
            article_ratings = []
            for uid in self.trainset.all_users():
                try:
                    pred = self.model.predict(uid, article_id)
                    article_ratings.append(pred.est)
                except:
                    continue
            return np.mean(article_ratings) if article_ratings else 3.0
        
        # Known user, new article  
        elif user_id in self.trainset.all_users() and article_id not in self.trainset.all_items():
            # Return user's average rating
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]['avg_rating']
            return 3.0
        
        # Both new
        else:
            return 3.0  # Global average
    
    def get_user_recommendations(self, 
                               user_id: int, 
                               n_recommendations: int = 100,
                               exclude_seen: bool = True) -> List[Dict]:
        """Get recommendations for a specific user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Handle cold start (new user)
        if user_id not in self.trainset.all_users():
            return self._get_cold_start_recommendations(user_id, n_recommendations)
        
        # Get items the user has already interacted with
        seen_articles = set()
        if exclude_seen and user_id in self.user_item_matrix.index:
            seen_articles = set(
                self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0]
            )
        
        # Generate predictions for all articles
        recommendations = []
        for article_id in self.trainset.all_items():
            if exclude_seen and article_id in seen_articles:
                continue
                
            predicted_rating = self.predict(user_id, article_id)
            
            recommendations.append({
                'article_id': article_id,
                'cf_score': predicted_rating,
                'method': 'collaborative_filtering'
            })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['cf_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_cold_start_recommendations(self, 
                                      user_id: int, 
                                      n_recommendations: int) -> List[Dict]:
        """Get recommendations for new users (cold start)."""
        logger.info(f"Generating cold start recommendations for user {user_id}")
        
        recommendations = []
        for article_id in self.popular_articles[:n_recommendations * 2]:  # Get more to filter
            # Estimate rating based on article popularity
            if article_id in self.trainset.all_items():
                # Use average rating from similar users
                avg_rating = 3.0
                try:
                    # Sample some users and get their ratings for this article
                    sample_users = np.random.choice(
                        list(self.trainset.all_users()), 
                        min(10, len(self.trainset.all_users())), 
                        replace=False
                    )
                    ratings = []
                    for sample_user in sample_users:
                        pred = self.model.predict(sample_user, article_id)
                        ratings.append(pred.est)
                    avg_rating = np.mean(ratings)
                except:
                    avg_rating = 3.0
            else:
                avg_rating = 3.0
            
            recommendations.append({
                'article_id': article_id,
                'cf_score': avg_rating,
                'method': 'cold_start_popular'
            })
        
        recommendations.sort(key=lambda x: x['cf_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find users similar to the given user."""
        if not self.is_fitted or user_id not in self.trainset.all_users():
            return []
        
        # Get user's rating vector
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate similarities with all other users
        similarities = []
        for other_user in self.user_item_matrix.index:
            if other_user == user_id:
                continue
                
            other_ratings = self.user_item_matrix.loc[other_user]
            
            # Calculate cosine similarity on commonly rated items
            common_items = (user_ratings > 0) & (other_ratings > 0)
            if common_items.sum() < 2:  # Need at least 2 common items
                continue
                
            user_common = user_ratings[common_items]
            other_common = other_ratings[common_items]
            
            # Cosine similarity
            similarity = np.dot(user_common, other_common) / (
                np.linalg.norm(user_common) * np.linalg.norm(other_common)
            )
            
            similarities.append((other_user, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]
    
    def get_article_recommendations_for_user_set(self, 
                                               user_ids: List[int], 
                                               n_recommendations: int = 50) -> List[Dict]:
        """Get article recommendations that would appeal to a set of users."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get predictions for all articles for all users
        article_scores = defaultdict(list)
        
        for user_id in user_ids:
            if user_id in self.trainset.all_users():
                for article_id in self.trainset.all_items():
                    score = self.predict(user_id, article_id)
                    article_scores[article_id].append(score)
        
        # Calculate aggregate scores for each article
        recommendations = []
        for article_id, scores in article_scores.items():
            if len(scores) >= len(user_ids) * 0.5:  # At least half the users have scores
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Consider both average appeal and consensus (low std)
                consensus_score = avg_score * (1 - std_score / 5.0)  # Penalize high variance
                
                recommendations.append({
                    'article_id': article_id,
                    'avg_score': avg_score,
                    'consensus_score': consensus_score,
                    'score_std': std_score,
                    'n_users_scored': len(scores)
                })
        
        recommendations.sort(key=lambda x: x['consensus_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def evaluate_model(self, test_df: pd.DataFrame = None, cv_folds: int = 3) -> Dict:
        """Evaluate the collaborative filtering model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        results = {}
        
        if test_df is not None:
            # Evaluate on provided test set
            reader = Reader(rating_scale=(1, 5))
            test_data = Dataset.load_from_df(test_df[['user_id', 'article_id', 'rating']], reader)
            testset = test_data.build_full_trainset().build_testset()
            
            predictions = self.model.test(testset)
            
            results['test_rmse'] = accuracy.rmse(predictions, verbose=False)
            results['test_mae'] = accuracy.mae(predictions, verbose=False)
        
        # Cross-validation on training data
        if cv_folds > 1:
            # Reconstruct dataset for CV
            # This is a simplified approach - in practice, you'd want to reconstruct from original data
            results['cv_folds'] = cv_folds
            logger.info("CV evaluation would require original dataset reconstruction")
        
        return results
    
    def save_model(self, model_name: str = "cf_surprise_model") -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_path = self.model_dir / f"{model_name}.pkl"
        
        model_data = {
            'model': self.model,
            'trainset': self.trainset,
            'user_item_matrix': self.user_item_matrix,
            'user_ids': self.user_ids,
            'article_ids': self.article_ids,
            'user_profiles': self.user_profiles,
            'popular_articles': self.popular_articles,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"CF model saved to {model_path}")
    
    def load_model(self, model_name: str = "cf_surprise_model") -> 'CollaborativeFilteringModel':
        """Load a fitted model from disk."""
        model_path = self.model_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.trainset = model_data['trainset']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_ids = model_data['user_ids']
        self.article_ids = model_data['article_ids']
        self.user_profiles = model_data['user_profiles']
        self.popular_articles = model_data['popular_articles']
        self.config = model_data['config']
        self.is_fitted = True
        
        logger.info(f"CF model loaded from {model_path}")
        return self
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'n_users': self.trainset.n_users,
            'n_items': self.trainset.n_items,
            'n_ratings': self.trainset.n_ratings,
            'rating_scale': (self.trainset.rating_scale[0], self.trainset.rating_scale[1]),
            'sparsity': 1 - (self.trainset.n_ratings / (self.trainset.n_users * self.trainset.n_items)),
            'n_factors': self.config['n_factors'],
            'config': self.config
        }
    
    def get_user_statistics(self, user_id: int) -> Dict:
        """Get statistics for a specific user."""
        if not self.is_fitted or user_id not in self.trainset.all_users():
            return {'exists': False}
        
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0]
            
            return {
                'exists': True,
                'n_ratings': len(rated_items),
                'avg_rating': rated_items.mean(),
                'rating_std': rated_items.std(),
                'most_liked_articles': rated_items.nlargest(5).index.tolist()
            }
        
        return {'exists': True, 'n_ratings': 0}


def create_interaction_ratings(interactions_df: pd.DataFrame, 
                             rating_weights: dict = None) -> pd.DataFrame:
    """Create explicit ratings from implicit interactions."""
    if rating_weights is None:
        rating_weights = {
            'like': 5.0,
            'read': 4.0, 
            'click': 3.0,
            'share': 5.0,
            'dislike': 1.0
        }
    
    # Map events to ratings
    interactions_df['rating'] = interactions_df['event'].map(rating_weights).fillna(3.0)
    
    # Aggregate multiple interactions (e.g., if user clicked and then liked)
    user_item_ratings = interactions_df.groupby(['user_id', 'article_id']).agg({
        'rating': 'max',  # Take the highest rating if multiple interactions
        'ts': 'max'       # Most recent timestamp
    }).reset_index()
    
    return user_item_ratings
