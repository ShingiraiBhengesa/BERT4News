"""Flask web application for the news recommendation system."""
import os
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import pandas as pd
import json

from recsys.models.hybrid import HybridRecommender
from recsys.db import NewsDatabase
from recsys.config import FLASK_CONFIG, TOPICS, RECOMMENDATION_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)
CORS(app)

# Global variables for models and database
hybrid_model = None
db = None

def init_app():
    """Initialize the application with models and database."""
    global hybrid_model, db
    
    try:
        # Initialize database
        db = NewsDatabase()
        logger.info("Database initialized")
        
        # Try to load trained models
        hybrid_model = HybridRecommender()
        try:
            hybrid_model.load_model("hybrid_recommender")
            logger.info("Loaded trained hybrid model")
        except FileNotFoundError:
            logger.warning("No trained model found - recommendations will use fallback methods")
            hybrid_model = None
        
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
        # Continue without models for basic functionality


@app.route('/')
def index():
    """Home page with topic selection for new users."""
    return render_template('index.html', topics=TOPICS)


@app.route('/onboarding', methods=['POST'])
def onboarding():
    """Handle user onboarding - topic selection."""
    try:
        selected_topics = request.json.get('topics', [])
        diversity_preference = request.json.get('diversity', 0.5)
        allow_new_sources = request.json.get('allow_new_sources', True)
        
        # Validate topics
        valid_topics = [topic for topic in selected_topics if topic in TOPICS]
        if not valid_topics:
            return jsonify({'error': 'Please select at least one valid topic'}), 400
        
        # Store preferences in session
        session['user_topics'] = valid_topics
        session['diversity_preference'] = diversity_preference
        session['allow_new_sources'] = allow_new_sources
        session['user_id'] = None  # New user
        
        return jsonify({
            'success': True,
            'message': 'Preferences saved successfully',
            'redirect': '/recommendations'
        })
        
    except Exception as e:
        logger.error(f"Error in onboarding: {e}")
        return jsonify({'error': 'Failed to save preferences'}), 500


@app.route('/recommendations')
def recommendations():
    """Show personalized recommendations."""
    user_topics = session.get('user_topics', ['technology', 'politics'])
    diversity_preference = session.get('diversity_preference', 0.1)
    
    return render_template(
        'recommendations.html',
        user_topics=user_topics,
        diversity_preference=diversity_preference
    )


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for getting recommendations."""
    try:
        data = request.json
        user_id = data.get('user_id') or session.get('user_id')
        k = data.get('k', 10)
        diversity = data.get('diversity', 0.1)
        allow_new_sources = data.get('allow_new_sources', True)
        
        recommendations = []
        
        if user_id and hybrid_model and hybrid_model.is_fitted:
            # Existing user with trained model
            try:
                recommendations = hybrid_model.recommend(
                    user_id=user_id,
                    n_recommendations=k,
                    diversity_weight=diversity,
                    include_explanations=True
                )
            except Exception as e:
                logger.warning(f"Hybrid model failed for user {user_id}: {e}")
                recommendations = []
        
        # Fallback for new users or if model failed
        if not recommendations:
            user_topics = session.get('user_topics', ['technology', 'politics'])
            if hybrid_model and hybrid_model.is_fitted:
                try:
                    recommendations = hybrid_model.recommend_cold_start(
                        user_topics=user_topics,
                        n_recommendations=k
                    )
                except Exception as e:
                    logger.warning(f"Cold start recommendations failed: {e}")
                    recommendations = get_fallback_recommendations(user_topics, k)
            else:
                recommendations = get_fallback_recommendations(user_topics, k)
        
        # Convert to API format
        api_recommendations = []
        for rec in recommendations:
            article_info = get_article_details(rec['article_id'])
            if article_info:
                api_rec = {
                    'article_id': rec['article_id'],
                    'title': article_info.get('title', 'No title'),
                    'summary': article_info.get('summary', 'No summary available'),
                    'source': article_info.get('source', 'Unknown'),
                    'published_at': article_info.get('published_at', ''),
                    'topics': article_info.get('topics', ''),
                    'url': article_info.get('url', ''),
                    'score': rec.get('hybrid_score', rec.get('final_score', 0.5)),
                    'explanation': rec.get('explanation', 'Recommended for you'),
                    'method': rec.get('method', 'hybrid')
                }
                api_recommendations.append(api_rec)
        
        return jsonify({
            'recommendations': api_recommendations,
            'count': len(api_recommendations),
            'user_id': user_id,
            'fallback_used': not (user_id and hybrid_model and hybrid_model.is_fitted)
        })
        
    except Exception as e:
        logger.error(f"Error in recommend API: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for recording user feedback."""
    try:
        data = request.json
        user_id = data.get('user_id') or session.get('user_id')
        article_id = data.get('article_id')
        event = data.get('event')  # like, dislike, click, read, hide_source, more_like_this
        
        if not article_id or not event:
            return jsonify({'error': 'Missing article_id or event'}), 400
        
        # Create user if doesn't exist
        if not user_id:
            user_id = create_new_user(session.get('user_topics', []))
            session['user_id'] = user_id
        
        # Record interaction in database
        interaction_data = {
            'user_id': user_id,
            'article_id': article_id,
            'event': event,
            'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dwell_time_s': data.get('dwell_time', 0)
        }
        
        df = pd.DataFrame([interaction_data])
        db.insert_interactions(df)
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded',
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500


@app.route('/api/article/<int:article_id>')
def api_article_details(article_id):
    """API endpoint to get detailed article information."""
    try:
        article_info = get_article_details(article_id)
        if not article_info:
            return jsonify({'error': 'Article not found'}), 404
        
        return jsonify(article_info)
        
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {e}")
        return jsonify({'error': 'Failed to fetch article'}), 500


@app.route('/profile')
def profile():
    """User profile page."""
    user_id = session.get('user_id')
    user_topics = session.get('user_topics', [])
    
    # Get user statistics if available
    user_stats = {}
    if user_id and hybrid_model and hybrid_model.is_fitted:
        try:
            user_stats = hybrid_model.cf_model.get_user_statistics(user_id)
        except:
            pass
    
    return render_template(
        'profile.html',
        user_id=user_id,
        user_topics=user_topics,
        user_stats=user_stats,
        all_topics=TOPICS
    )


@app.route('/api/update_profile', methods=['POST'])
def api_update_profile():
    """Update user profile preferences."""
    try:
        data = request.json
        new_topics = data.get('topics', [])
        diversity_preference = data.get('diversity', 0.1)
        
        # Validate topics
        valid_topics = [topic for topic in new_topics if topic in TOPICS]
        if not valid_topics:
            return jsonify({'error': 'Please select at least one valid topic'}), 400
        
        # Update session
        session['user_topics'] = valid_topics
        session['diversity_preference'] = diversity_preference
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({'error': 'Failed to update profile'}), 500


def get_article_details(article_id):
    """Get article details from database."""
    try:
        query = "SELECT * FROM articles WHERE article_id = ?"
        with db.get_connection() as conn:
            result = conn.execute(query, [article_id]).fetchone()
        
        if result:
            return dict(result)
        return None
        
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {e}")
        return None


def get_fallback_recommendations(user_topics, k=10):
    """Get fallback recommendations when models are not available."""
    try:
        # Get popular recent articles
        popular_articles = db.get_popular_articles(days=7, limit=k*2)
        
        recommendations = []
        for _, article in popular_articles.iterrows():
            article_topics = article.get('topics', '').split(',')
            article_topics = [t.strip().lower() for t in article_topics if t.strip()]
            
            # Simple topic matching
            topic_match = any(topic.lower() in article_topics for topic in user_topics)
            score = 0.8 if topic_match else 0.5
            
            recommendations.append({
                'article_id': article['article_id'],
                'final_score': score,
                'method': 'fallback_popular',
                'explanation': 'Popular recent article'
            })
        
        # Sort by score and return top k
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return recommendations[:k]
        
    except Exception as e:
        logger.error(f"Error getting fallback recommendations: {e}")
        return []


def create_new_user(topics):
    """Create a new user in the database."""
    try:
        # Find the next user ID
        query = "SELECT MAX(user_id) as max_id FROM users"
        with db.get_connection() as conn:
            result = conn.execute(query).fetchone()
            next_id = (result['max_id'] or 0) + 1
        
        # Create user record
        user_data = {
            'user_id': next_id,
            'signup_ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'declared_topics': ','.join(topics),
            'profile_data': json.dumps({'created_via': 'web_app'})
        }
        
        df = pd.DataFrame([user_data])
        db.insert_users(df)
        
        logger.info(f"Created new user {next_id}")
        return next_id
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return None


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error"), 500


if __name__ == '__main__':
    init_app()
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
