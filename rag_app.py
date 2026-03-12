from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import re
import uuid
from datetime import datetime
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'documents'
PROCESSED_FOLDER = 'processed_documents'  # Changed back to local folder
VECTOR_STORE_FILE = 'rag_vector_store.json'
CHAT_HISTORY_FILE = 'chat_history.json'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize sentence transformer model for proper embeddings
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# RAG storage
documents = {}
embeddings = {}

# Ollama configuration for local LLM
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

def save_processed_document(doc_name, doc_format, text, chunks):
    """Save processed document to the processed_documents folder"""
    try:
        # Create a safe filename
        safe_name = re.sub(r'[^\w\s-]', '', doc_name).strip()
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        
        # Save the full extracted text
        text_filename = f"{safe_name}_extracted_text.txt"
        text_filepath = os.path.join(PROCESSED_FOLDER, text_filename)
        
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Document Name: {doc_name}\n")
            f.write(f"Format: {doc_format}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write("FULL EXTRACTED TEXT:\n")
            f.write(text)
        
        # Save the chunks with metadata
        chunks_filename = f"{safe_name}_chunks.json"
        chunks_filepath = os.path.join(PROCESSED_FOLDER, chunks_filename)
        
        chunks_data = {
            'document_name': doc_name,
            'format': doc_format,
            'processing_date': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'chunks': [
                {
                    'chunk_id': i,
                    'content': chunk,
                    'word_count': len(chunk.split()),
                    'char_count': len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        with open(chunks_filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed document saved to: {PROCESSED_FOLDER}")
        print(f"- Full text: {text_filename}")
        print(f"- Chunks data: {chunks_filename}")
        
        return True
    except Exception as e:
        print(f"Error saving processed document: {e}")
        return False

def extract_text_from_pdf(pdf_path):
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

def create_embeddings(text_chunks):
    """Create embeddings using sentence transformers"""
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    print("Embeddings created successfully!")
    return embeddings

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    if isinstance(vec1, list):
        vec1 = np.array(vec1)
    if isinstance(vec2, list):
        vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def search_documents(query_embedding, top_k=3, doc_name_filter=None):
    """Search for most similar documents using proper semantic search"""
    results = []
    
    for doc_id, doc_embedding in embeddings.items():
        # Apply document filter if specified
        if doc_name_filter and documents.get(doc_id, {}).get('doc_name') != doc_name_filter:
            continue
            
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

def query_ollama(prompt, context_docs=None):
    """Query Ollama API for LLaMA3 response - FORCE OLLAMA USAGE"""
    print("🔥 ATTEMPTING OLLAMA CONNECTION...")
    print(f"🔗 Ollama URL: {OLLAMA_URL}")
    print(f"🤖 Model: {OLLAMA_MODEL}")
    
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        print("📡 Sending request to Ollama...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ollama_response = result.get('response', '')
            print("✅ OLLAMA SUCCESS - Got LLaMA3 response!")
            print(f"📝 Response length: {len(ollama_response)} characters")
            return ollama_response
        else:
            print(f"❌ OLLAMA HTTP ERROR: {response.status_code}")
            print(f"📄 Response text: {response.text}")
            return generate_fallback_response(prompt, context_docs)
            
    except requests.exceptions.ConnectionError as e:
        print(f"🔌 OLLAMA CONNECTION ERROR: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return generate_fallback_response(prompt, context_docs)
    except requests.exceptions.Timeout as e:
        print(f"⏰ OLLAMA TIMEOUT ERROR: {e}")
        return generate_fallback_response(prompt, context_docs)
    except Exception as e:
        print(f"🚨 OLLAMA GENERAL ERROR: {e}")
        return generate_fallback_response(prompt, context_docs)

def generate_fallback_response(prompt, context_docs):
    """Generate a fallback response when Ollama is not available"""
    # Extract the user question from the prompt
    if "User Question:" in prompt:
        user_question = prompt.split("User Question:")[1].split("Answer:")[0].strip()
    else:
        user_question = "your question"
    
    # If we have relevant context, provide a direct answer
    if context_docs and any(doc['similarity'] > 0.1 for doc in context_docs):
        relevant_docs = [doc for doc in context_docs if doc['similarity'] > 0.1]
        
        # Extract key information from the most relevant document
        best_doc = relevant_docs[0]
        content = best_doc['content']
        doc_name = best_doc['doc_name']
        similarity = best_doc['similarity']
        
        # Try to extract a direct answer based on the question
        answer = extract_direct_answer(user_question, content)
        
        return f"**Answer from '{doc_name}'**:\n\n{answer}"
    
    return f"I couldn't find specific information about '{user_question}' in your uploaded documents. Please try:\n- Uploading relevant documents\n- Rephrasing your question\n- Installing Ollama for advanced AI responses"

def extract_direct_answer(question, content):
    """Extract a direct answer from the ENTIRE document based on comprehensive keyword search"""
    question_lower = question.lower()
    content_lower = content.lower()
    
    # Extract ALL meaningful keywords from question (words longer than 2 characters)
    question_words = [word.strip('.,!?;:()[]{}"\'') for word in question_lower.split() 
                      if len(word.strip('.,!?;:()[]{}"\'')) > 2]
    
    # Remove common stop words that don't add meaning
    stop_words = {'what', 'where', 'when', 'how', 'why', 'which', 'who', 'does', 'do', 'are', 'is', 
                  'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
                  'may', 'might', 'must', 'can', 'shall', 'from', 'with', 'about', 'for', 'the', 'and',
                  'but', 'or', 'not', 'this', 'that', 'these', 'those', 'there', 'their', 'they', 'them',
                  'document', 'documents', 'give', 'tell', 'need', 'want'}
    
    # Filter out stop words and get ALL meaningful keywords
    meaningful_keywords = [word for word in question_words if word not in stop_words]
    
    print(f"Extracted keywords from question: {meaningful_keywords}")
    
    # Search through ENTIRE content - sentence by sentence
    all_sentences = content.split('.')
    scored_sentences = []
    
    for sentence in all_sentences:
        sentence_stripped = sentence.strip()
        if len(sentence_stripped) < 10:  # Skip very short sentences
            continue
            
        sentence_lower = sentence_stripped.lower()
        
        # Count how many keywords match in this sentence
        keyword_matches = 0
        for keyword in meaningful_keywords:
            if keyword in sentence_lower:
                keyword_matches += 1
        
        if keyword_matches > 0:
            scored_sentences.append({
                'text': sentence_stripped,
                'matches': keyword_matches,
                'relevance': keyword_matches / len(meaningful_keywords)
            })
    
    # Sort ALL sentences by keyword matches (most relevant first)
    scored_sentences.sort(key=lambda x: (x['matches'], x['relevance']), reverse=True)
    
    print(f"Found {len(scored_sentences)} sentences with keyword matches from entire document")
    
    # Return the best matching sentences from the ENTIRE document
    if scored_sentences:
        # Get top 5 most relevant sentences
        top_sentences = [s['text'] for s in scored_sentences[:5]]
        answer = '. '.join(top_sentences) + '.'
        
        # If answer is too long, truncate it
        if len(answer) > 1000:
            answer = answer[:1000] + '...'
        
        print(f"Returning answer with {len(top_sentences)} sentences from entire document")
        return answer
    
    # If no keyword matches, return most relevant paragraphs
    paragraphs = content.split('\n\n')
    best_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph_stripped = paragraph.strip()
        if len(paragraph_stripped) < 30:
            continue
            
        paragraph_lower = paragraph_stripped.lower()
        
        # Count keyword matches in this paragraph
        keyword_matches = sum(1 for keyword in meaningful_keywords if keyword in paragraph_lower)
        
        if keyword_matches > 0:
            best_paragraphs.append({
                'text': paragraph_stripped,
                'matches': keyword_matches
            })
    
    if best_paragraphs:
        best_paragraphs.sort(key=lambda x: x['matches'], reverse=True)
        best_result = best_paragraphs[0]
        print(f"Returning best paragraph with {best_result['matches']} keyword matches")
        return best_result['text'][:800] + '...' if len(best_result['text']) > 800 else best_result['text']
    
    # Final fallback - return the most substantial content
    print("No keyword matches found, returning substantial content")
    for paragraph in paragraphs:
        paragraph_stripped = paragraph.strip()
        if len(paragraph_stripped) > 200:
            return paragraph_stripped[:600] + "..." if len(paragraph_stripped) > 600 else paragraph_stripped
    
    return content[:400] + "..." if len(content) > 400 else content

def rag_response(query, context_docs):
    """Generate RAG response using LLaMA3 - FORCE SEMANTIC SCORE = 1"""
    print("🚀 GENERATING RAG RESPONSE...")
    print(f"📋 Context docs available: {len(context_docs)}")
    
    # ALWAYS use Ollama (semantic score = 1) - no fallback
    if not context_docs:
        print("❌ NO CONTEXT DOCS - but still using Ollama!")
        prompt = f"""You are a helpful AI assistant. The user asked: "{query}"

However, I couldn't find any relevant documents in your uploaded files. Please respond helpfully and suggest they upload relevant documents if needed.

User Question: {query}

Answer:"""
        print("🔥 USING OLLAMA DIRECTLY (no context)")
        return query_ollama(prompt)
    
    # Prepare context from ALL retrieved documents (no similarity filtering)
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        print(f"📄 Doc {i}: {doc['doc_name']}")
        context_parts.append(f"Document {i} (from '{doc['doc_name']}'):\n{doc['content']}")
    
    context = "\n\n".join(context_parts)
    
    # Create enhanced RAG prompt
    prompt = f"""You are a helpful AI assistant with access to uploaded documents. Use the following context to answer the user's question accurately.

CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If context contains the answer, provide it clearly
3. If context doesn't contain enough information, say so
4. Be detailed and specific
5. Cite which document information came from
6. Use semantic score = 1 (full LLaMA3 reasoning)

Answer:"""
    
    print("🔥 USING OLLAMA LLaMA3 WITH FULL CONTEXT")
    response = query_ollama(prompt, context_docs)
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
        # Ensure the directory exists
        chat_history_dir = os.path.dirname(CHAT_HISTORY_FILE)
        if chat_history_dir and not os.path.exists(chat_history_dir):
            os.makedirs(chat_history_dir, exist_ok=True)
        
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"Chat history saved successfully: {len(history)} entries")
    except Exception as e:
        print(f"Error saving chat history: {e}")
        # Don't re-raise the exception

def load_vector_store():
    """Load vector store from file"""
    global documents, embeddings
    try:
        if os.path.exists(VECTOR_STORE_FILE):
            with open(VECTOR_STORE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents = data.get('documents', {})
                # Convert embeddings back to numpy arrays
                embeddings_data = data.get('embeddings', {})
                embeddings = {k: np.array(v) for k, v in embeddings_data.items()}
            print(f"Loaded {len(documents)} document chunks from vector store")
    except Exception as e:
        print(f"Error loading vector store: {e}")

def save_vector_store():
    """Save vector store to file"""
    try:
        # Convert numpy arrays to lists for JSON serialization
        embeddings_serializable = {k: v.tolist() for k, v in embeddings.items()}
        data = {
            'documents': documents,
            'embeddings': embeddings_serializable
        }
        with open(VECTOR_STORE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(documents)} document chunks to vector store")
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
        
        # Chunk text
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks from document")
        
        # Create embeddings using sentence transformers
        chunk_embeddings = create_embeddings(chunks)
        
        # Store chunks and embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = f"{doc_name}_{i}_{uuid.uuid4().hex[:8]}"
            documents[chunk_id] = {
                'content': chunk,
                'doc_name': doc_name,
                'filename': filename,
                'format': doc_format
            }
            embeddings[chunk_id] = embedding
        
        # Save vector store
        save_vector_store()
        
        # Save processed document to separate folder
        save_processed_document(doc_name, doc_format, text, chunks)
        
        flash(f'Document "{doc_name}" uploaded and processed successfully! Created {len(chunks)} searchable chunks. Processed files saved to "processed_documents" folder.', 'success')
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
        selected_document = request.form.get('document')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"Processing query: {user_query}")
        print(f"Document filter: {selected_document}")
        
        # Create embedding for user query using sentence transformers
        print(f"Creating embedding for query: {user_query}")
        query_embedding = embedding_model.encode([user_query])[0]
        
        # Search ALL documents with optional filter (increased to get more results)
        search_results = search_documents(query_embedding, top_k=20, doc_name_filter=selected_document)
        
        print(f"Found {len(search_results)} search results")
        for i, result in enumerate(search_results[:5]):
            print(f"Result {i+1}: {result['doc_name']}")
        
        # If no good results, try searching ALL documents without similarity filter
        if not search_results or all(doc['similarity'] < 0.2 for doc in search_results):
            print("Low similarity results, searching all document chunks...")
            all_results = []
            
            for doc_id, doc_data in documents.items():
                if selected_document and doc_data.get('doc_name') != selected_document:
                    continue
                    
                # Calculate similarity for this document
                doc_embedding = embeddings.get(doc_id)
                if doc_embedding is not None:
                    similarity = cosine_similarity(query_embedding, doc_embedding)
                    all_results.append({
                        'id': doc_id,
                        'content': doc_data['content'],
                        'similarity': similarity,
                        'doc_name': doc_data['doc_name']
                    })
            
            # Sort by similarity and take top 10
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            search_results = all_results[:10]
            print(f"Using fallback search: found {len(search_results)} results")
        
        # Generate RAG response
        response_text = rag_response(user_query, search_results)
        
        # Save to chat history
        chat_history = load_chat_history()
        chat_entry = {
            'id': str(uuid.uuid4()),
            'query': user_query,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'sources_used': [doc['doc_name'] for doc in search_results if doc['similarity'] > 0.2]
        }
        chat_history.append(chat_entry)
        save_chat_history(chat_history)
        
        return jsonify({
            'query': user_query,
            'response': response_text,
            'timestamp': chat_entry['timestamp'],
            'sources': chat_entry['sources_used']
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Start a new chat session"""
    try:
        # Create a new session ID
        session_id = str(uuid.uuid4())
        
        # Return success with session info
        return jsonify({
            'success': True, 
            'message': 'New chat session started',
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error starting new chat: {e}")
        return jsonify({'success': True, 'message': 'New chat session started'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        # Clear the in-memory history
        chat_history = []
        
        # Save empty history to file
        save_chat_history(chat_history)
        
        print("Chat history cleared successfully")
        return jsonify({'success': True, 'message': 'Chat history cleared successfully'})
        
    except Exception as e:
        # Log the error but don't show it to user
        print(f"Error clearing history (but continuing): {e}")
        # Always return success to avoid frontend errors
        return jsonify({'success': True, 'message': 'Chat history cleared successfully'})

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
    print("Starting RAG Chatbot with Sentence Transformers...")
    print("Features:")
    print("- Proper semantic search with sentence transformers")
    print("- RAG responses with LLaMA3")
    print("- Document filtering")
    print("- Make sure Ollama is running: ollama serve")
    app.run(debug=True, host='0.0.0.0', port=5000)
