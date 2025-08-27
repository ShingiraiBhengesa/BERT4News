/**
 * Main JavaScript functionality for BERT4News
 */

// Global variables
let currentUser = null;
let notificationTimeout = null;

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Set up global error handling
    window.addEventListener('error', function(e) {
        console.error('Global error:', e.error);
        showAlert('An unexpected error occurred. Please try again.', 'error');
    });

    // Set up unhandled promise rejection handling
    window.addEventListener('unhandledrejection', function(e) {
        console.error('Unhandled promise rejection:', e.reason);
        showAlert('An unexpected error occurred. Please try again.', 'error');
        e.preventDefault();
    });
}

/**
 * Show alert message to user
 * @param {string} message - Alert message
 * @param {string} type - Alert type: success, error, warning, info
 * @param {number} duration - Duration in ms (default: 5000)
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;

    // Clear existing notification timeout
    if (notificationTimeout) {
        clearTimeout(notificationTimeout);
    }

    // Map alert types to Bootstrap classes
    const alertClasses = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    };

    const alertClass = alertClasses[type] || alertClasses['info'];
    
    // Get appropriate icon
    const icons = {
        'success': 'fas fa-check-circle',
        'error': 'fas fa-exclamation-circle',
        'warning': 'fas fa-exclamation-triangle',
        'info': 'fas fa-info-circle'
    };

    const icon = icons[type] || icons['info'];

    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
            <i class="${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;

    alertContainer.innerHTML = alertHtml;

    // Auto-dismiss after duration
    if (duration > 0) {
        notificationTimeout = setTimeout(() => {
            const alert = alertContainer.querySelector('.alert');
            if (alert) {
                const bootstrapAlert = new bootstrap.Alert(alert);
                bootstrapAlert.close();
            }
        }, duration);
    }

    // Scroll to top to ensure alert is visible
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Make API request with error handling
 * @param {string} url - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise} Response promise
 */
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

/**
 * Debounce function to limit rapid function calls
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @param {boolean} immediate - Execute immediately
 * @returns {Function} Debounced function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Throttle function to limit function calls
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in ms
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Format date for display
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date string
 */
function formatDate(date) {
    if (!date) return 'Unknown';
    
    try {
        const dateObj = new Date(date);
        const now = new Date();
        const diffTime = Math.abs(now - dateObj);
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
        const diffMinutes = Math.floor(diffTime / (1000 * 60));

        if (diffMinutes < 60) {
            return `${diffMinutes} minutes ago`;
        } else if (diffHours < 24) {
            return `${diffHours} hours ago`;
        } else if (diffDays < 7) {
            return `${diffDays} days ago`;
        } else {
            return dateObj.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }
    } catch (error) {
        console.error('Error formatting date:', error);
        return 'Unknown';
    }
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) {
        return text || '';
    }
    return text.substring(0, maxLength).trim() + '...';
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Generate topic icon based on topic name
 * @param {string} topic - Topic name
 * @returns {string} Font Awesome icon class
 */
function getTopicIcon(topic) {
    const icons = {
        'technology': 'fas fa-laptop',
        'politics': 'fas fa-landmark',
        'business': 'fas fa-briefcase',
        'sports': 'fas fa-futbol',
        'health': 'fas fa-heartbeat',
        'science': 'fas fa-flask',
        'entertainment': 'fas fa-film',
        'finance': 'fas fa-chart-line',
        'world': 'fas fa-globe',
        'lifestyle': 'fas fa-leaf'
    };
    return icons[topic.toLowerCase()] || 'fas fa-newspaper';
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showAlert('Copied to clipboard!', 'success', 2000);
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showAlert('Copied to clipboard!', 'success', 2000);
        } catch (fallbackError) {
            showAlert('Failed to copy to clipboard', 'error');
        }
        
        document.body.removeChild(textArea);
    }
}

/**
 * Share article via Web Share API or fallback
 * @param {object} article - Article data
 */
async function shareArticle(article) {
    const shareData = {
        title: article.title,
        text: article.summary,
        url: article.url || window.location.href
    };

    try {
        if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
            await navigator.share(shareData);
        } else {
            // Fallback: copy link to clipboard
            await copyToClipboard(shareData.url);
        }
    } catch (error) {
        console.error('Error sharing:', error);
        showAlert('Failed to share article', 'error');
    }
}

/**
 * Load more content with pagination
 * @param {string} endpoint - API endpoint
 * @param {object} params - Request parameters
 * @param {Function} renderFunction - Function to render results
 * @param {string} containerId - Container element ID
 */
async function loadMoreContent(endpoint, params, renderFunction, containerId) {
    const container = document.getElementById(containerId);
    const loadMoreBtn = document.getElementById('loadMoreBtn');
    
    if (loadMoreBtn) {
        loadMoreBtn.disabled = true;
        loadMoreBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    }

    try {
        const data = await apiRequest(endpoint, {
            method: 'POST',
            body: JSON.stringify(params)
        });

        if (data && data.results) {
            renderFunction(data.results, true); // append = true
            
            // Hide load more button if no more results
            if (loadMoreBtn && (!data.has_more || data.results.length === 0)) {
                loadMoreBtn.style.display = 'none';
            }
        }
    } catch (error) {
        showAlert('Failed to load more content', 'error');
    } finally {
        if (loadMoreBtn) {
            loadMoreBtn.disabled = false;
            loadMoreBtn.innerHTML = '<i class="fas fa-plus me-1"></i>Load More';
        }
    }
}

/**
 * Initialize search functionality
 */
function initializeSearch() {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) return;

    const debouncedSearch = debounce(performSearch, 300);
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        if (query.length >= 2) {
            debouncedSearch(query);
        } else {
            clearSearchResults();
        }
    });
}

/**
 * Perform search
 * @param {string} query - Search query
 */
async function performSearch(query) {
    const searchResults = document.getElementById('searchResults');
    if (!searchResults) return;

    try {
        searchResults.innerHTML = '<div class="text-center py-3"><div class="spinner-border spinner-border-sm text-primary"></div></div>';
        
        const data = await apiRequest('/api/search', {
            method: 'POST',
            body: JSON.stringify({ query, limit: 5 })
        });

        if (data.results && data.results.length > 0) {
            renderSearchResults(data.results);
        } else {
            searchResults.innerHTML = '<div class="text-muted text-center py-3">No results found</div>';
        }
    } catch (error) {
        searchResults.innerHTML = '<div class="text-danger text-center py-3">Search failed</div>';
    }
}

/**
 * Clear search results
 */
function clearSearchResults() {
    const searchResults = document.getElementById('searchResults');
    if (searchResults) {
        searchResults.innerHTML = '';
    }
}

/**
 * Render search results
 * @param {Array} results - Search results
 */
function renderSearchResults(results) {
    const searchResults = document.getElementById('searchResults');
    if (!searchResults) return;

    const html = results.map(article => `
        <div class="border-bottom py-2">
            <h6 class="mb-1">${escapeHtml(article.title)}</h6>
            <small class="text-muted">${escapeHtml(article.source)} • ${formatDate(article.published_at)}</small>
        </div>
    `).join('');

    searchResults.innerHTML = html;
}

/**
 * Track user interaction for analytics
 * @param {string} event - Event type
 * @param {object} data - Event data
 */
function trackInteraction(event, data) {
    // Simple analytics tracking - would integrate with actual analytics service
    if (typeof gtag !== 'undefined') {
        gtag('event', event, data);
    }
    
    // Log for debugging
    console.log('Interaction tracked:', event, data);
}

/**
 * Initialize keyboard shortcuts
 */
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('searchInput');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // R for refresh recommendations
        if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
            if (typeof refreshRecommendations === 'function') {
                e.preventDefault();
                refreshRecommendations();
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
}

/**
 * Initialize scroll to top functionality
 */
function initializeScrollToTop() {
    const scrollBtn = document.getElementById('scrollToTop');
    if (!scrollBtn) return;

    window.addEventListener('scroll', throttle(() => {
        if (window.pageYOffset > 300) {
            scrollBtn.style.display = 'block';
        } else {
            scrollBtn.style.display = 'none';
        }
    }, 100));

    scrollBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

/**
 * Save user preferences to localStorage
 * @param {object} preferences - User preferences
 */
function saveUserPreferences(preferences) {
    try {
        localStorage.setItem('bert4news_preferences', JSON.stringify(preferences));
    } catch (error) {
        console.error('Failed to save preferences:', error);
    }
}

/**
 * Load user preferences from localStorage
 * @returns {object} User preferences
 */
function loadUserPreferences() {
    try {
        const saved = localStorage.getItem('bert4news_preferences');
        return saved ? JSON.parse(saved) : {};
    } catch (error) {
        console.error('Failed to load preferences:', error);
        return {};
    }
}

/**
 * Check if user prefers dark mode
 * @returns {boolean} True if dark mode preferred
 */
function prefersDarkMode() {
    const saved = loadUserPreferences();
    if (saved.darkMode !== undefined) {
        return saved.darkMode;
    }
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/**
 * Apply theme based on user preference
 */
function applyTheme() {
    const isDark = prefersDarkMode();
    document.body.classList.toggle('dark-theme', isDark);
    
    // Update theme toggle button if it exists
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.innerHTML = isDark ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
    }
}

/**
 * Toggle theme
 */
function toggleTheme() {
    const currentPrefs = loadUserPreferences();
    const newDarkMode = !prefersDarkMode();
    
    saveUserPreferences({ ...currentPrefs, darkMode: newDarkMode });
    applyTheme();
    
    showAlert(`Switched to ${newDarkMode ? 'dark' : 'light'} theme`, 'success', 2000);
}

// Initialize components when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeSearch();
    initializeKeyboardShortcuts();
    initializeScrollToTop();
    applyTheme();
});

// Export functions for use in other scripts
window.BERT4News = {
    showAlert,
    apiRequest,
    formatDate,
    truncateText,
    escapeHtml,
    getTopicIcon,
    copyToClipboard,
    shareArticle,
    trackInteraction,
    saveUserPreferences,
    loadUserPreferences,
    toggleTheme
};
