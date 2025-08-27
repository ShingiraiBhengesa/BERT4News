"""Comprehensive evaluation metrics for recommendation systems."""
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Union
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score
import warnings

logger = logging.getLogger(__name__)


class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommendation systems."""
    
    def __init__(self, k_values: List[int] = None):
        """Initialize with evaluation parameters.
        
        Args:
            k_values: List of k values for top-k metrics (default: [5, 10, 20])
        """
        self.k_values = k_values or [5, 10, 20]
        self.results = {}
        
    def evaluate_recommendations(self,
                               recommendations: Dict[int, List[int]],
                               ground_truth: Dict[int, List[int]],
                               user_item_matrix: pd.DataFrame = None) -> Dict:
        """Evaluate recommendations against ground truth.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended article_ids
            ground_truth: Dict mapping user_id to list of relevant article_ids
            user_item_matrix: User-item matrix for additional metrics
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Calculate relevance metrics
        for k in self.k_values:
            results[f'precision@{k}'] = self.calculate_precision_at_k(
                recommendations, ground_truth, k
            )
            results[f'recall@{k}'] = self.calculate_recall_at_k(
                recommendations, ground_truth, k
            )
            results[f'f1@{k}'] = self.calculate_f1_at_k(
                recommendations, ground_truth, k
            )
            results[f'ndcg@{k}'] = self.calculate_ndcg_at_k(
                recommendations, ground_truth, k
            )
        
        # Calculate diversity metrics
        if user_item_matrix is not None:
            results['intra_list_diversity'] = self.calculate_intra_list_diversity(
                recommendations, user_item_matrix
            )
            results['coverage'] = self.calculate_coverage(
                recommendations, user_item_matrix
            )
        
        # Calculate novelty and serendipity
        results['novelty'] = self.calculate_novelty(recommendations, user_item_matrix)
        results['catalog_coverage'] = self.calculate_catalog_coverage(
            recommendations, user_item_matrix
        )
        
        # Calculate popularity bias
        results['popularity_bias'] = self.calculate_popularity_bias(
            recommendations, user_item_matrix
        )
        
        self.results = results
        return results
    
    def calculate_precision_at_k(self,
                                recommendations: Dict[int, List[int]],
                                ground_truth: Dict[int, List[int]],
                                k: int) -> float:
        """Calculate Precision@K across all users."""
        precisions = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
                
            recommended_k = recommendations[user_id][:k]
            relevant_items = set(ground_truth[user_id])
            
            if len(recommended_k) == 0:
                precisions.append(0.0)
                continue
                
            hits = len(set(recommended_k) & relevant_items)
            precision = hits / len(recommended_k)
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def calculate_recall_at_k(self,
                             recommendations: Dict[int, List[int]],
                             ground_truth: Dict[int, List[int]],
                             k: int) -> float:
        """Calculate Recall@K across all users."""
        recalls = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
                
            recommended_k = recommendations[user_id][:k]
            relevant_items = set(ground_truth[user_id])
            
            if len(relevant_items) == 0:
                continue
                
            hits = len(set(recommended_k) & relevant_items)
            recall = hits / len(relevant_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def calculate_f1_at_k(self,
                         recommendations: Dict[int, List[int]],
                         ground_truth: Dict[int, List[int]],
                         k: int) -> float:
        """Calculate F1@K across all users."""
        precision = self.calculate_precision_at_k(recommendations, ground_truth, k)
        recall = self.calculate_recall_at_k(recommendations, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_ndcg_at_k(self,
                           recommendations: Dict[int, List[int]],
                           ground_truth: Dict[int, List[int]],
                           k: int) -> float:
        """Calculate NDCG@K across all users."""
        ndcg_scores = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
                
            recommended_k = recommendations[user_id][:k]
            relevant_items = set(ground_truth[user_id])
            
            if len(relevant_items) == 0 or len(recommended_k) == 0:
                continue
            
            # Create relevance scores (1 for relevant, 0 for not relevant)
            y_true = [1 if item in relevant_items else 0 for item in recommended_k]
            y_score = list(range(len(recommended_k), 0, -1))  # Decreasing scores by rank
            
            if sum(y_true) == 0:  # No relevant items in recommendations
                ndcg_scores.append(0.0)
                continue
            
            try:
                # Reshape for sklearn
                y_true_reshaped = np.array([y_true])
                y_score_reshaped = np.array([y_score])
                
                ndcg = ndcg_score(y_true_reshaped, y_score_reshaped, k=k)
                ndcg_scores.append(ndcg)
            except Exception as e:
                logger.warning(f"NDCG calculation failed for user {user_id}: {e}")
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def calculate_intra_list_diversity(self,
                                     recommendations: Dict[int, List[int]],
                                     user_item_matrix: pd.DataFrame,
                                     similarity_threshold: float = 0.7) -> float:
        """Calculate intra-list diversity (1 - avg pairwise similarity)."""
        diversities = []
        
        for user_id, rec_list in recommendations.items():
            if len(rec_list) < 2:
                diversities.append(1.0)  # Single item is maximally diverse
                continue
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item1, item2 = rec_list[i], rec_list[j]
                    
                    # Simple Jaccard similarity based on common users
                    if item1 in user_item_matrix.columns and item2 in user_item_matrix.columns:
                        users1 = set(user_item_matrix[user_item_matrix[item1] > 0].index)
                        users2 = set(user_item_matrix[user_item_matrix[item2] > 0].index)
                        
                        if len(users1) == 0 and len(users2) == 0:
                            similarity = 0.0
                        else:
                            intersection = len(users1 & users2)
                            union = len(users1 | users2)
                            similarity = intersection / union if union > 0 else 0.0
                    else:
                        similarity = 0.0  # Items not in matrix are considered dissimilar
                    
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            diversity = 1.0 - avg_similarity
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def calculate_coverage(self,
                          recommendations: Dict[int, List[int]],
                          user_item_matrix: pd.DataFrame) -> float:
        """Calculate catalog coverage (fraction of items recommended)."""
        all_items = set(user_item_matrix.columns) if user_item_matrix is not None else set()
        recommended_items = set()
        
        for rec_list in recommendations.values():
            recommended_items.update(rec_list)
        
        if len(all_items) == 0:
            return 0.0
        
        return len(recommended_items & all_items) / len(all_items)
    
    def calculate_novelty(self,
                         recommendations: Dict[int, List[int]],
                         user_item_matrix: pd.DataFrame = None) -> float:
        """Calculate novelty as 1/popularity of recommended items."""
        if user_item_matrix is None:
            return 0.0
        
        # Calculate item popularity (number of users who interacted with each item)
        item_popularity = (user_item_matrix > 0).sum(axis=0)
        total_users = len(user_item_matrix)
        
        novelties = []
        
        for rec_list in recommendations.values():
            item_novelties = []
            for item in rec_list:
                if item in item_popularity.index:
                    pop_score = item_popularity[item] / total_users
                    novelty = -np.log2(pop_score) if pop_score > 0 else 0
                    item_novelties.append(novelty)
            
            if item_novelties:
                novelties.append(np.mean(item_novelties))
        
        return np.mean(novelties) if novelties else 0.0
    
    def calculate_catalog_coverage(self,
                                  recommendations: Dict[int, List[int]],
                                  user_item_matrix: pd.DataFrame = None) -> float:
        """Calculate what fraction of the catalog is being recommended."""
        if user_item_matrix is None:
            return 0.0
        
        all_items = set(user_item_matrix.columns)
        recommended_items = set()
        
        for rec_list in recommendations.values():
            recommended_items.update(rec_list)
        
        return len(recommended_items) / len(all_items) if all_items else 0.0
    
    def calculate_popularity_bias(self,
                                 recommendations: Dict[int, List[int]],
                                 user_item_matrix: pd.DataFrame = None) -> float:
        """Calculate popularity bias (Gini coefficient of item recommendations)."""
        if user_item_matrix is None:
            return 0.0
        
        # Count how many times each item was recommended
        item_rec_counts = defaultdict(int)
        total_recommendations = 0
        
        for rec_list in recommendations.values():
            for item in rec_list:
                item_rec_counts[item] += 1
                total_recommendations += 1
        
        if total_recommendations == 0:
            return 0.0
        
        # Calculate Gini coefficient
        recommendation_counts = list(item_rec_counts.values())
        recommendation_counts.sort()
        
        n = len(recommendation_counts)
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(recommendation_counts)
        gini = (n + 1 - 2 * np.sum(cumsum)) / (n * np.sum(recommendation_counts))
        
        return gini
    
    def calculate_serendipity(self,
                             recommendations: Dict[int, List[int]],
                             user_profiles: Dict[int, List[int]],
                             relevance_threshold: float = 0.5) -> float:
        """Calculate serendipity (relevant but unexpected recommendations)."""
        serendipity_scores = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in user_profiles:
                continue
            
            user_history = set(user_profiles[user_id])
            unexpected_relevant = 0
            total_relevant = 0
            
            for item in rec_list:
                # Simple heuristic: item is unexpected if not in user's history
                is_unexpected = item not in user_history
                # In a real system, you'd have relevance scores
                is_relevant = True  # Placeholder - would need actual relevance data
                
                if is_relevant:
                    total_relevant += 1
                    if is_unexpected:
                        unexpected_relevant += 1
            
            if total_relevant > 0:
                serendipity = unexpected_relevant / total_relevant
                serendipity_scores.append(serendipity)
        
        return np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    def generate_evaluation_report(self, 
                                 save_path: Optional[str] = None,
                                 include_plots: bool = True) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return "No evaluation results available. Run evaluate_recommendations first."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Relevance metrics
        report_lines.append("RELEVANCE METRICS:")
        report_lines.append("-" * 30)
        for k in self.k_values:
            precision = self.results.get(f'precision@{k}', 0)
            recall = self.results.get(f'recall@{k}', 0)
            f1 = self.results.get(f'f1@{k}', 0)
            ndcg = self.results.get(f'ndcg@{k}', 0)
            
            report_lines.append(f"k={k:2d}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, NDCG={ndcg:.4f}")
        
        report_lines.append("")
        
        # Diversity and coverage metrics
        report_lines.append("DIVERSITY & COVERAGE METRICS:")
        report_lines.append("-" * 35)
        
        diversity = self.results.get('intra_list_diversity', 0)
        coverage = self.results.get('coverage', 0)
        novelty = self.results.get('novelty', 0)
        catalog_coverage = self.results.get('catalog_coverage', 0)
        popularity_bias = self.results.get('popularity_bias', 0)
        
        report_lines.append(f"Intra-list Diversity: {diversity:.4f}")
        report_lines.append(f"Coverage:            {coverage:.4f}")
        report_lines.append(f"Novelty:             {novelty:.4f}")
        report_lines.append(f"Catalog Coverage:    {catalog_coverage:.4f}")
        report_lines.append(f"Popularity Bias:     {popularity_bias:.4f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        # Generate plots if requested
        if include_plots:
            self.plot_metrics(save_path)
        
        return report
    
    def plot_metrics(self, save_prefix: Optional[str] = None):
        """Generate visualization plots for evaluation metrics."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Plot relevance metrics by k
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['precision', 'recall', 'f1', 'ndcg']
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            values = []
            for k in self.k_values:
                values.append(self.results.get(f'{metric}@{k}', 0))
            
            ax.plot(self.k_values, values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('k')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()}@k')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_prefix:
            plt.savefig(f"{save_prefix}_relevance_metrics.png", dpi=150, bbox_inches='tight')
        
        # Plot diversity metrics
        diversity_metrics = {
            'Intra-list Diversity': self.results.get('intra_list_diversity', 0),
            'Coverage': self.results.get('coverage', 0),
            'Novelty': self.results.get('novelty', 0),
            'Catalog Coverage': self.results.get('catalog_coverage', 0),
            'Popularity Bias': self.results.get('popularity_bias', 0)
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(diversity_metrics.keys(), diversity_metrics.values())
        plt.title('Diversity and Coverage Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        
        # Color bars based on whether higher is better
        colors = ['green', 'green', 'green', 'green', 'red']  # Red for popularity bias (lower is better)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.tight_layout()
        
        if save_prefix:
            plt.savefig(f"{save_prefix}_diversity_metrics.png", dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def compare_models(self, 
                      model_results: Dict[str, Dict],
                      save_path: Optional[str] = None) -> pd.DataFrame:
        """Compare multiple models' evaluation results."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            row = {'Model': model_name}
            
            # Add relevance metrics
            for k in self.k_values:
                for metric in ['precision', 'recall', 'f1', 'ndcg']:
                    row[f'{metric}@{k}'] = results.get(f'{metric}@{k}', 0)
            
            # Add diversity metrics
            diversity_metrics = [
                'intra_list_diversity', 'coverage', 'novelty', 
                'catalog_coverage', 'popularity_bias'
            ]
            for metric in diversity_metrics:
                row[metric] = results.get(metric, 0)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Model comparison saved to {save_path}")
        
        return df


# Standalone functions for individual metrics
def calculate_precision_at_k(recommendations: Dict[int, List[int]],
                           ground_truth: Dict[int, List[int]],
                           k: int) -> float:
    """Calculate Precision@K."""
    evaluator = RecommendationMetrics()
    return evaluator.calculate_precision_at_k(recommendations, ground_truth, k)


def calculate_recall_at_k(recommendations: Dict[int, List[int]],
                        ground_truth: Dict[int, List[int]],
                        k: int) -> float:
    """Calculate Recall@K."""
    evaluator = RecommendationMetrics()
    return evaluator.calculate_recall_at_k(recommendations, ground_truth, k)


def calculate_ndcg_at_k(recommendations: Dict[int, List[int]],
                       ground_truth: Dict[int, List[int]],
                       k: int) -> float:
    """Calculate NDCG@K."""
    evaluator = RecommendationMetrics()
    return evaluator.calculate_ndcg_at_k(recommendations, ground_truth, k)


def create_test_train_split_temporal(interactions_df: pd.DataFrame,
                                   test_days: int = 7,
                                   min_interactions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create temporal train/test split for evaluation."""
    # Sort by timestamp
    interactions_sorted = interactions_df.sort_values('ts')
    
    # Find split point (last N days for test)
    max_timestamp = pd.to_datetime(interactions_sorted['ts'].max())
    split_timestamp = max_timestamp - pd.Timedelta(days=test_days)
    
    # Split data
    train_df = interactions_sorted[
        pd.to_datetime(interactions_sorted['ts']) < split_timestamp
    ].copy()
    
    test_df = interactions_sorted[
        pd.to_datetime(interactions_sorted['ts']) >= split_timestamp
    ].copy()
    
    # Filter users with minimum interactions in training set
    user_counts = train_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    train_df = train_df[train_df['user_id'].isin(valid_users)]
    test_df = test_df[test_df['user_id'].isin(valid_users)]
    
    logger.info(f"Train set: {len(train_df)} interactions, {train_df['user_id'].nunique()} users")
    logger.info(f"Test set: {len(test_df)} interactions, {test_df['user_id'].nunique()} users")
    
    return train_df, test_df
