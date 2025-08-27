"""Text processing utilities for news articles."""
import re
import string
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class TextProcessor:
    """Advanced text processing for news articles."""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        
        # Add custom stop words for news
        self.news_stop_words = {
            'said', 'says', 'according', 'reported', 'report', 'news',
            'article', 'story', 'post', 'blog', 'website', 'site',
            'today', 'yesterday', 'tomorrow', 'monday', 'tuesday', 
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        }
        self.stop_words.update(self.news_stop_words)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def tokenize_and_filter(self, text: str, min_length: int = 2) -> List[str]:
        """Tokenize text and filter out stop words and short words."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text, language=self.language)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if it's a stop word, punctuation, or too short
            if (token not in self.stop_words and 
                token not in string.punctuation and
                len(token) >= min_length and
                token.isalpha()):  # Only alphabetic tokens
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text using simple frequency analysis."""
        tokens = self.tokenize_and_filter(text)
        
        if not tokens:
            return []
        
        # Count frequency
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def preprocess_for_tfidf(self, text: str) -> str:
        """Preprocess text specifically for TF-IDF vectorization."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_filter(cleaned)
        return ' '.join(tokens)
    
    def extract_sentences(self, text: str, max_sentences: int = 3) -> List[str]:
        """Extract first few sentences for summary."""
        if not text:
            return []
        
        sentences = sent_tokenize(text, language=self.language)
        return sentences[:max_sentences]
    
    def calculate_readability_score(self, text: str) -> float:
        """Calculate a simple readability score (0-1, higher is more readable)."""
        if not text:
            return 0.0
        
        sentences = sent_tokenize(text, language=self.language)
        words = word_tokenize(text, language=self.language)
        
        if not sentences or not words:
            return 0.0
        
        # Simple metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Normalize scores (these are rough heuristics)
        sentence_score = max(0, min(1, 1 - (avg_sentence_length - 15) / 20))
        word_score = max(0, min(1, 1 - (avg_word_length - 5) / 5))
        
        return (sentence_score + word_score) / 2
    
    def detect_language(self, text: str) -> str:
        """Simple language detection (placeholder - could use langdetect library)."""
        # For now, assume English - could be enhanced with actual language detection
        return 'english'


def clean_text(text: str) -> str:
    """Standalone function for quick text cleaning."""
    processor = TextProcessor()
    return processor.clean_text(text)


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Standalone function for keyword extraction."""
    processor = TextProcessor()
    return processor.extract_keywords(text, top_k)


def preprocess_articles_batch(articles_df, text_columns=['title', 'summary', 'content']):
    """Batch preprocessing of articles DataFrame."""
    processor = TextProcessor()
    
    # Combine text columns
    articles_df = articles_df.copy()
    combined_text = []
    
    for _, row in articles_df.iterrows():
        text_parts = []
        for col in text_columns:
            if col in row and row[col] and isinstance(row[col], str):
                text_parts.append(row[col])
        
        combined = ' '.join(text_parts)
        cleaned = processor.preprocess_for_tfidf(combined)
        combined_text.append(cleaned)
    
    articles_df['processed_text'] = combined_text
    
    # Extract keywords for each article
    articles_df['keywords'] = articles_df['processed_text'].apply(
        lambda x: extract_keywords(x, top_k=15)
    )
    
    logger.info(f"Preprocessed {len(articles_df)} articles")
    return articles_df


def build_vocabulary(texts: List[str], max_features: int = 10000) -> Dict[str, int]:
    """Build vocabulary from a list of texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    vectorizer.fit(texts)
    return vectorizer.vocabulary_


def extract_topic_keywords(text: str, topics: List[str]) -> Dict[str, List[str]]:
    """Extract keywords related to specific topics from text."""
    topic_keywords = {topic: [] for topic in topics}
    
    # Simple keyword matching - could be enhanced with more sophisticated NLP
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    
    # Define topic-related keywords (this could be loaded from a config file)
    topic_mappings = {
        'technology': ['tech', 'ai', 'artificial intelligence', 'machine learning', 
                      'software', 'hardware', 'computer', 'internet', 'digital'],
        'politics': ['political', 'government', 'election', 'vote', 'policy', 
                    'senate', 'congress', 'president', 'minister', 'law'],
        'business': ['business', 'company', 'corporate', 'finance', 'economy', 
                    'market', 'stock', 'revenue', 'profit', 'investment'],
        'sports': ['sport', 'game', 'team', 'player', 'match', 'championship', 
                  'tournament', 'score', 'win', 'loss'],
        'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 
                  'treatment', 'medicine', 'patient', 'wellness', 'fitness']
    }
    
    for topic in topics:
        if topic.lower() in topic_mappings:
            topic_words = topic_mappings[topic.lower()]
            for word in topic_words:
                if word in text_lower:
                    # Find related words in the vicinity
                    for i, token in enumerate(tokens):
                        if word in token or token in word:
                            # Get surrounding words
                            start = max(0, i - 2)
                            end = min(len(tokens), i + 3)
                            context_words = tokens[start:end]
                            topic_keywords[topic].extend(context_words)
    
    # Remove duplicates and stop words
    processor = TextProcessor()
    for topic in topic_keywords:
        topic_keywords[topic] = list(set([
            word for word in topic_keywords[topic] 
            if word not in processor.stop_words and len(word) > 2
        ]))
    
    return topic_keywords
