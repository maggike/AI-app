from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PyPDF2 import PdfReader
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
from transformers import GPT2TokenizerFast
from concurrent.futures import ThreadPoolExecutor
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding, EmbeddingModel
from sqlalchemy import text


import difflib
import re
import logging
import time
import openai
import threading

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)
UPLOAD_FOLDER = 'uploaded_docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
rate_limit_lock = threading.Lock()
requests_per_minute = 20
request_interval = 60 / requests_per_minute
last_request_time = 0 

load_dotenv()

sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

client = OpenAI(
    #api_key=os.environ.get("OPENAI_API_KEY"),
    organization='org-xn78pOCSNt4pW6XXtD528lib'
      # This is the default and can be omitted
)
# class ComparisonResult(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     semantic_similarity = db.Column(db.Float, nullable=False)
#     added_sections = db.Column(db.Text, nullable=False)
#     removed_sections = db.Column(db.Text, nullable=False)

#     def __init__(self, semantic_similarity, added_sections, removed_sections):
#         self.semantic_similarity = semantic_similarity
#         self.added_sections = added_sections
#         self.removed_sections = removed_sections
class ComparisonResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    semantic_similarity = db.Column(db.Float, nullable=False)
    added_sections = db.Column(db.Text, nullable=False)
    removed_sections = db.Column(db.Text, nullable=False)
    risks = db.Column(db.Text, nullable=True)  # Add the risks field

    def __init__(self, semantic_similarity, added_sections, removed_sections, risks=None):
        self.semantic_similarity = semantic_similarity
        self.added_sections = added_sections
        self.removed_sections = removed_sections
        self.risks = risks

# Initialize the database
with app.app_context():
    db.create_all()

# Save results to database
@app.route('/api/save_results', methods=['POST'])
def save_results():
    data = request.json
    result = ComparisonResult(
        semantic_similarity=data['semantic_similarity'],
        added_sections='\n'.join(data['added_sections']),
        removed_sections='\n'.join(data['removed_sections']),
        risks='\n'.join(data['risks']) if 'risks' in data else None
    )
    db.session.add(result)
    db.session.commit()
    return jsonify({'message': 'Results saved successfully', 'id': result.id})
@app.route('/api/test_db', methods=['GET'])
def test_db():
    try:
        result = db.session.execute(text("SELECT 1")).scalar()
        return jsonify({'message': 'Database connected successfully', 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def safe_request_with_limit(api_call, *args, **kwargs):
    """
    Handles API calls with centralized rate-limiting and retries.
    """
    global last_request_time
    max_retries = 5
    backoff_factor = 2
    retry_delay = 1  # Start with a 1-second delay

    for attempt in range(max_retries):
        elapsed = time.time() - last_request_time
        if elapsed < request_interval:
            time.sleep(request_interval - elapsed)

        try:
            response = api_call(*args, **kwargs)
            last_request_time = time.time()
            return response
        except openai.RateLimitError as e:
            logging.warning(f"Rate limit error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= backoff_factor
        except (openai.APIError) as e:
            logging.warning(f"API error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= backoff_factor
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break
    raise Exception("Max retries reached. Could not complete the request.")
def compare_structures(text1, text2):
    sections1 = extract_sections(text1)
    sections2 = extract_sections(text2)
    added_sections = [s for s in sections2 if s not in sections1]
    removed_sections = [s for s in sections1 if s not in sections2]

    # Summarize structural changes
    summary = {
        "added_count": len(added_sections),
        "removed_count": len(removed_sections),
        "critical_added": [s for s in added_sections if "liability" in s.lower()],
        "critical_removed": [s for s in removed_sections if "liability" in s.lower()]
    }
    return summary


def split_text_into_chunks(text, max_tokens=1024, overlap=128):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks


def get_embeddings_batch(text_chunks):
    try:
        batch_size = 5
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            with rate_limit_lock:
                time.sleep(request_interval)  # Enforce rate limiting
            response = safe_request_with_limit(
                client.embeddings.create,
                input=batch,
                model="text-embedding-ada-002"
            )
            all_embeddings.extend([data.embedding for data in response.data])
        return all_embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return []
def shorten_risks(risks, max_length=200):
    """
    Shorten risk descriptions to a maximum character length.
    """
    return [risk if len(risk) <= max_length else risk[:max_length] + "..." for risk in risks]
def filter_risks_by_keywords(risks, keywords):
    """
    Filter risks that contain specific keywords.
    """
    filtered_risks = [risk for risk in risks if any(keyword.lower() in risk.lower() for keyword in keywords)]
    return filtered_risks
def summarize_risks(risks):
    """
    Summarize a list of risks into concise statements.
    """
    risks_text = "\n".join(risks)
    try:
        response = safe_request_with_limit(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal and financial summarizer."},
                {"role": "user", "content": f"Summarize the following risks concisely: {risks_text}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing risks: {e}"

def analyze_and_refine_risks(differences):
    raw_risks = assess_risk_parallel(differences)  # Generate raw risks
    #filtered_risks = deduplicate_risks(raw_risks)  # Remove redundancy
    concise_risks = shorten_risks(filtered_risks)  # Shorten long descriptions
    keywords = ["financial", "compliance", "merger"]  # Context-specific filtering
    relevant_risks = filter_risks_by_keywords(concise_risks, keywords)  # Filter by keywords
    summarized_risks = summarize_risks(relevant_risks)  # Summarize risks for clarity
    return summarized_risks

def calculate_similarity(text1, text2):
    try:
        # Split large texts into manageable chunks
        max_tokens = 1024  # Safe limit below the model's 8192 token max
        logging.info("Splitting text1 into chunks...")
        text1_chunks = split_text_into_chunks(text1, max_tokens)
        logging.info(f"Text1 split into {len(text1_chunks)} chunks.")
        logging.info("Splitting text2 into chunks...")
        text2_chunks = split_text_into_chunks(text2, max_tokens)
        logging.info(f"Text2 split into {len(text2_chunks)} chunks.")


       
        logging.info("Generating embeddings for text1...")
        embeddings1 = get_embeddings_batch(text1_chunks)
        logging.info(f"Generated {len(embeddings1)} embeddings for text1.")
        logging.info("Generating embeddings for text2...")
        embeddings2 = get_embeddings_batch(text2_chunks)
        logging.info(f"Generated {len(embeddings2)} embeddings for text2.")
        embeddings1 = [emb for emb in embeddings1 if emb is not None]
        embeddings2 = [emb for emb in embeddings2 if emb is not None]

       

        if not embeddings1 or not embeddings2:
            raise ValueError("Failed to generate embeddings for one or both texts.")

        # Average embeddings for each text
        logging.info("Calculating cosine similarity...")
        avg_embedding1 = np.mean(embeddings1, axis=0)
        avg_embedding2 = np.mean(embeddings2, axis=0)

        # Compute cosine similarity
        similarity = cosine_similarity([avg_embedding1], [avg_embedding2])[0][0]
        logging.info(f"Similarity calculated: {similarity:.4f}")
        return similarity
    except Exception as e:
        print(f"Error in calculating similarity: {e}")
        return None


def assess_risk_parallel(differences):
    max_tokens = 512
    chunks = [split_text_into_chunks(diff, max_tokens) for diff in differences]
    chunks_flat = [chunk for sublist in chunks for chunk in sublist]

    def analyze_chunk(chunk):
        try:
            response = safe_request_with_limit(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal and financial risk analyst."},
                    {"role": "user", "content": f"Identify risks or opportunities in the following change: {chunk}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error analyzing chunk: {e}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        risks = list(executor.map(analyze_chunk, chunks_flat))

    unique_risks = list(set(risks))
    return unique_risks

def summarize_text(text):
    try:
        response = safe_request(client.chat.completions.create,
            model="gpt-3.5-turbo",  # Or "gpt-3.5-turbo" for faster responses
            messages=[
                {"role": "system", "content": "You are a helpful assistant for summarizing documents."},
                {"role": "user", "content": f"Summarize the following text: {text}"}
            ],
            max_tokens=200,
            temperature=0.5
        )
        logging.debug(f"API Response: {response}")
       
        summary = response.choices[0].message.content.strip()
        logging.debug(f"Generated Summary: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Summarization error: {str(e)}")
        return f"Error in summarization: {str(e)}"
def compare_texts(text1, text2):
    
    diff = list(difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm=''))
    filtered_diff = [line for line in diff if re.search(r'\d+|[A-Z]{2,}|risk|liability', line, re.IGNORECASE)]
    return '\n'.join(filtered_diff)

def extract_sections(text):
    # Extract sections based on a common pattern (e.g., numbered headings)
    sections = re.findall(r'^\d+\.\s.+$', text, flags=re.MULTILINE)
    return sections



def extract_text(file_path):
    try:
        # Handle PDFs
        if file_path.endswith('.pdf'):
            text = ''
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
            return text.strip()

        # Handle DOCX
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return '\n'.join([p.text for p in doc.paragraphs])

        # Handle images (JPG, PNG) using OCR
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)

        # Unsupported file type
        else:
            return 'Unsupported file type for text extraction.'

    except Exception as e:
        return f'Error extracting text: {str(e)}'


@app.route('/api/compare_documents', methods=['POST'])
def compare_documents():
    try: 
        # Ensure both files are uploaded
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Two files are required for comparison'}), 400

        # Save the uploaded files
        file1 = request.files['file1']
        file2 = request.files['file2']
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        # Extract text from both files
        text1 = extract_text(file1_path)
        text2 = extract_text(file2_path)
        textual_diff = compare_texts(text1, text2)
        semantic_similarity = calculate_similarity(text1, text2)
        
        structure_summary = compare_structures(text1, text2)
        added_sections = structure_summary["critical_added"]
        removed_sections = structure_summary["critical_removed"]
        risks = assess_risk_parallel(textual_diff.splitlines())
        
        return jsonify({
            'message': 'Documents compared successfully.',
            'textual_diff': textual_diff,
            'semantic_similarity': semantic_similarity,
            'added_sections': added_sections,
            'removed_sections': removed_sections,
            'risks': risks
        })
    except Exception as e:
        # Log the detailed error
        logging.error(f"Error processing files: {str(e)}", exc_info=True)

        # Return detailed error for debugging (use only in dev, avoid in production)
        return jsonify({'error': f"Error processing files: {str(e)}"}), 500
@app.route('/api/extract_and_summarize', methods=['POST'])
def extract_and_summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract text
    extracted_text = extract_text(file_path)

    # Summarize text
    summary = summarize_text(extracted_text)

    return jsonify({'message': 'Text extracted and summarized successfully', 
                    'text': extracted_text, 
                    'summary': summary})

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)
