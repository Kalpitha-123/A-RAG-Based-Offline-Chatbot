from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import re
import uuid
from datetime import datetime
import math
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'documents'
VECTOR_STORE_FILE = 'simple_vector_store.json'
CHAT_HISTORY_FILE = 'chat_history.json'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory vector storage
documents = {}
embeddings = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def extract_text_from_html(html_path):
    """Extract text from HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error extracting HTML: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into chunks with overlap"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= text_length:
            break
            
        start = end - overlap
    
    return chunks

def simple_embedding(text):
    """Create a simple TF-IDF like embedding"""
    # Simple word frequency based embedding
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create a fixed-size vector (first 100 dimensions)
    vector = []
    words_list = list(word_freq.keys())[:100]
    for word in words_list:
        vector.append(word_freq[word])
    
    # Pad or truncate to 100 dimensions
    while len(vector) < 100:
        vector.append(0)
    
    return vector[:100]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def search_documents(query_embedding, top_k=3):
    """Search for most similar documents"""
    results = []
    
    for doc_id, doc_embedding in embeddings.items():
        similarity = cosine_similarity(query_embedding, doc_embedding)
        if doc_id in documents:
            results.append({
                'id': doc_id,
                'content': documents[doc_id]['content'],
                'similarity': similarity,
                'doc_name': documents[doc_id]['doc_name']
            })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]

def simple_llm_response(query, context):
    """Generate a simple response based on context"""
    if not context:
        return "I couldn't find any relevant information in the uploaded documents. Please make sure you have uploaded documents related to your query."
    
    # Simple rule-based response generation
    context_text = " ".join([doc['content'] for doc in context])
    
    # Extract key information from context
    response = f"Based on the uploaded documents, here's what I found about '{query}':\n\n"
    
    for i, doc in enumerate(context, 1):
        if doc['similarity'] > 0.1:  # Only include relevant results
            response += f"{i}. From '{doc['doc_name']}':\n"
            response += f"   {doc['content'][:300]}...\n\n"
    
    if len(context) == 0 or all(doc['similarity'] <= 0.1 for doc in context):
        response = "I couldn't find specific information about your query in the uploaded documents. Try uploading more relevant documents or rephrasing your question."
    
    return response

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat history: {e}")
    return []

def save_chat_history(history):
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_vector_store():
    """Load vector store from file"""
    global documents, embeddings
    try:
        if os.path.exists(VECTOR_STORE_FILE):
            with open(VECTOR_STORE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents = data.get('documents', {})
                embeddings = data.get('embeddings', {})
    except Exception as e:
        print(f"Error loading vector store: {e}")

def save_vector_store():
    """Save vector store to file"""
    try:
        data = {
            'documents': documents,
            'embeddings': embeddings
        }
        with open(VECTOR_STORE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving vector store: {e}")

@app.route('/')
def index():
    return redirect(url_for('training'))

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        doc_name = request.form.get('doc_name')
        doc_format = request.form.get('doc_format')
        
        if not doc_name or not doc_format:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('training'))
        
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('training'))
        
        file = request.files['document']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('training'))
        
        # Save file
        filename = str(uuid.uuid4()) + '_' + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Extract text based on format
        if doc_format == 'pdf':
            text = extract_text_from_pdf(filepath)
        elif doc_format == 'html':
            text = extract_text_from_html(filepath)
        else:
            flash('Invalid document format', 'error')
            return redirect(url_for('training'))
        
        if not text:
            flash('Failed to extract text from document', 'error')
            return redirect(url_for('training'))
        
        # Chunk text and create embeddings
        chunks = chunk_text(text)
        
        # Store chunks and embeddings
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_name}_{i}_{uuid.uuid4().hex[:8]}"
            documents[chunk_id] = {
                'content': chunk,
                'doc_name': doc_name,
                'filename': filename,
                'format': doc_format
            }
            embeddings[chunk_id] = simple_embedding(chunk)
        
        # Save vector store
        save_vector_store()
        
        flash(f'Document "{doc_name}" uploaded and processed successfully!', 'success')
        return redirect(url_for('training'))
        
    except Exception as e:
        flash(f'Error processing document: {str(e)}', 'error')
        return redirect(url_for('training'))

@app.route('/chatbot')
def chatbot():
    chat_history = load_chat_history()
    return render_template('chatbot.html', chat_history=chat_history)

@app.route('/query', methods=['POST'])
def query_documents():
    try:
        user_query = request.form.get('query')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Create embedding for user query
        query_embedding = simple_embedding(user_query)
        
        # Search documents
        search_results = search_documents(query_embedding, top_k=3)
        
        # Generate response
        response_text = simple_llm_response(user_query, search_results)
        
        # Save to chat history
        chat_history = load_chat_history()
        chat_entry = {
            'id': str(uuid.uuid4()),
            'query': user_query,
            'response': response_text,
            'timestamp': datetime.now().isoformat()
        }
        chat_history.append(chat_entry)
        save_chat_history(chat_history)
        
        return jsonify({
            'query': user_query,
            'response': response_text,
            'timestamp': chat_entry['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        save_chat_history([])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Error clearing history: {str(e)}'}), 500

@app.route('/get_documents')
def get_documents():
    try:
        # Get unique documents from storage
        doc_names = set()
        for doc_data in documents.values():
            if 'doc_name' in doc_data:
                doc_names.add(doc_data['doc_name'])
        
        return jsonify({'documents': sorted(list(doc_names))})
    except Exception as e:
        return jsonify({'error': f'Error fetching documents: {str(e)}'}), 500

# Load vector store on startup
load_vector_store()

if __name__ == '__main__':
    print("Starting Simple Offline RAG Chatbot...")
    print("This version uses simple text matching instead of complex ML models")
    app.run(debug=True, host='0.0.0.0', port=5000)
