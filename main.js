// Main JavaScript file for Offline RAG Chatbot

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 fade-in ${
        type === 'error' ? 'bg-red-500 text-white' :
        type === 'success' ? 'bg-green-500 text-white' :
        type === 'warning' ? 'bg-yellow-500 text-white' :
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// File validation
function validateFile(file) {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const validTypes = ['application/pdf', 'text/html', 'text/htm'];
    const validExtensions = ['.pdf', '.html', '.htm'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (file.size > maxSize) {
        throw new Error('File size must be less than 50MB');
    }
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        throw new Error('Please upload a PDF or HTML file');
    }
    
    return true;
}

// Auto-detect document format
function detectDocumentFormat(file) {
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (file.type === 'application/pdf' || fileExtension === '.pdf') {
        return 'pdf';
    } else if (file.type === 'text/html' || fileExtension === '.html' || fileExtension === '.htm') {
        return 'html';
    }
    
    return null;
}

// API helpers
async function makeRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Chat utilities
function scrollToBottom(container) {
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator(container) {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'flex justify-end mb-4';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    container.appendChild(typingDiv);
    scrollToBottom(container);
    return typingDiv;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
}

function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex ${isUser ? 'justify-start' : 'justify-end'} mb-4 fade-in`;
    
    messageDiv.innerHTML = `
        <div class="message-bubble ${isUser ? 'bot-message' : 'user-message'} rounded-2xl px-4 py-3">
            <p class="text-sm font-medium ${isUser ? 'text-gray-600' : 'opacity-90'} mb-1">${isUser ? 'You:' : 'AI:'}</p>
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    return messageDiv;
}

// Document management
async function loadDocuments() {
    try {
        const data = await makeRequest('/get_documents');
        return data.documents || [];
    } catch (error) {
        console.error('Failed to load documents:', error);
        return [];
    }
}

function updateDocumentsList(documents) {
    const container = document.getElementById('documentsList');
    if (!container) return;
    
    if (documents.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center">No documents uploaded yet</p>';
        return;
    }
    
    container.innerHTML = documents.map(doc => `
        <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover-lift">
            <div class="flex items-center">
                <svg class="h-5 w-5 text-indigo-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
                <span class="text-gray-700">${escapeHtml(doc)}</span>
            </div>
            <span class="text-xs text-green-600 bg-green-100 px-2 py-1 rounded badge badge-success">Processed</span>
        </div>
    `).join('');
}

// Form validation
function validateForm(formData) {
    const errors = [];
    
    if (!formData.get('doc_name')?.trim()) {
        errors.push('Document name is required');
    }
    
    if (!formData.get('doc_format')) {
        errors.push('Document format is required');
    }
    
    if (!formData.get('document')?.size) {
        errors.push('Please select a file');
    }
    
    return errors;
}

// Initialize tooltips and other interactive elements
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'absolute z-10 px-2 py-1 text-sm text-white bg-gray-800 rounded shadow-lg';
            tooltip.textContent = this.getAttribute('data-tooltip');
            tooltip.style.bottom = '100%';
            tooltip.style.left = '50%';
            tooltip.style.transform = 'translateX(-50%)';
            tooltip.style.marginBottom = '5px';
            
            this.style.position = 'relative';
            this.appendChild(tooltip);
        });
        
        element.addEventListener('mouseleave', function() {
            const tooltip = this.querySelector('.absolute.z-10');
            if (tooltip) {
                tooltip.remove();
            }
        });
    });
}

// Keyboard shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('queryInput');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear input
        if (e.key === 'Escape') {
            const activeElement = document.activeElement;
            if (activeElement && activeElement.tagName === 'INPUT') {
                activeElement.value = '';
                activeElement.blur();
            }
        }
    });
}

// Theme management (optional)
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    initializeKeyboardShortcuts();
    initializeTheme();
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Export functions for use in other files
window.ChatbotUtils = {
    formatFileSize,
    formatTimestamp,
    escapeHtml,
    showNotification,
    validateFile,
    detectDocumentFormat,
    makeRequest,
    scrollToBottom,
    addTypingIndicator,
    removeTypingIndicator,
    createMessageElement,
    loadDocuments,
    updateDocumentsList,
    validateForm,
    toggleTheme
};
