from flask import Flask, render_template, request, jsonify
import re
import math
from collections import Counter
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

# BART model imports
try:
    from transformers import BartForConditionalGeneration, BartTokenizer
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    logging.warning("Transformers library not available. Only extractive summarization will work.")

app = Flask(__name__)

# Global variables for BART model
bart_model = None
bart_tokenizer = None


def extract_text_from_url(url):
    """Extract article text from a URL using web scraping"""
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None, "Invalid URL format"
        
        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script, style, nav, footer, header, aside elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript']):
            element.decompose()
        
        # Try to find article content using common patterns
        article_text = ""
        
        # Method 1: Look for article tag
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Method 2: Look for common article class names
        if not article_text or len(article_text) < 200:
            content_divs = soup.find_all(['div', 'section'], class_=re.compile(
                r'(article|content|post|story|entry|main|body|text)', re.I
            ))
            for div in content_divs:
                paragraphs = div.find_all('p')
                text = ' '.join([p.get_text().strip() for p in paragraphs])
                if len(text) > len(article_text):
                    article_text = text
        
        # Method 3: Fallback to all paragraphs
        if not article_text or len(article_text) < 200:
            paragraphs = soup.find_all('p')
            article_paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50]
            article_text = ' '.join(article_paragraphs)
        
        # Clean up the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        if len(article_text) < 100:
            return None, "Could not extract enough content from the URL"
        
        return article_text, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Failed to fetch URL: {str(e)}"
    except Exception as e:
        return None, f"Error extracting content: {str(e)}"


def initialize_bart_model():
    """Initialize BART model for abstractive summarization"""
    global bart_model, bart_tokenizer
    
    if not BART_AVAILABLE:
        return False
    
    try:
        if bart_model is None:
            logging.info("Loading BART model... This may take a few minutes on first run.")
            bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            logging.info("BART model loaded successfully!")
        return True
    except Exception as e:
        logging.error(f"Failed to load BART model: {str(e)}")
        return False


def summarize_abstractive(text, max_length=300, min_length=100):
    """
    Generate abstractive summary using BART model
    """
    global bart_model, bart_tokenizer
    
    if not BART_AVAILABLE:
        return "Abstractive summarization not available. Install transformers: pip install transformers torch"
    
    # Initialize model if needed
    if not initialize_bart_model():
        return "Failed to load BART model for abstractive summarization"
    
    try:
        # Preprocess text
        text = preprocess_text(text)
        
        # Truncate text if too long (BART has token limits), but keep more context
        if len(text) > 2000:
            text = text[:2000]
        
        # Tokenize input
        inputs = bart_tokenizer(text, 
                              return_tensors='pt', 
                              max_length=1024, 
                              truncation=True,
                              padding=True)
        
        # Generate summary with improved parameters for better coverage
        summary_ids = bart_model.generate(inputs.input_ids, 
                                        max_length=max_length,
                                        min_length=min_length,
                                        num_beams=6,  # Increased for better quality
                                        no_repeat_ngram_size=3,  # Allow more diverse content
                                        length_penalty=1.0,  # Balanced length preference
                                        early_stopping=True,
                                        do_sample=True,  # Add some randomness for diversity
                                        temperature=0.8)  # Control randomness
        
        # Decode summary
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
        
    except Exception as e:
        logging.error(f"Error in abstractive summarization: {str(e)}")
        return f"Error generating abstractive summary: {str(e)}"


# NLP-based Text Summarizer using TF-IDF and Sentence Scoring

def preprocess_text(text):
    """Clean and preprocess text"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    """Split text into sentences using regex"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def tokenize_words(text):
    """Tokenize text into words (lowercase, alphanumeric only)"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words

def get_stopwords():
    """Return a set of common English stopwords"""
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
        'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
        'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
        'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
        'won', 'wouldn', 'said', 'also', 'would', 'could', 'one', 'two', 'may'
    }
    return stopwords

def calculate_word_frequencies(words, stopwords):
    """Calculate word frequencies excluding stopwords"""
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    word_freq = Counter(filtered_words)
    
    # Normalize frequencies
    if word_freq:
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
    
    return word_freq

def calculate_tf(sentence, stopwords):
    """Calculate Term Frequency for a sentence"""
    words = tokenize_words(sentence)
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    word_count = Counter(filtered_words)
    total_words = len(filtered_words) if filtered_words else 1
    
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf

def calculate_idf(sentences, stopwords):
    """Calculate Inverse Document Frequency for all words"""
    num_sentences = len(sentences)
    word_doc_count = Counter()
    
    for sentence in sentences:
        words = set(tokenize_words(sentence))
        filtered_words = {w for w in words if w not in stopwords and len(w) > 2}
        word_doc_count.update(filtered_words)
    
    idf = {}
    for word, count in word_doc_count.items():
        idf[word] = math.log((num_sentences + 1) / (count + 1)) + 1
    
    return idf

def calculate_tfidf_score(sentence, idf, stopwords):
    """Calculate TF-IDF score for a sentence"""
    tf = calculate_tf(sentence, stopwords)
    score = 0
    for word, tf_value in tf.items():
        if word in idf:
            score += tf_value * idf[word]
    return score

def score_sentences(sentences, word_freq):
    """Score sentences based on word frequency"""
    sentence_scores = {}
    
    for i, sentence in enumerate(sentences):
        words = tokenize_words(sentence)
        score = 0
        word_count = 0
        
        for word in words:
            if word in word_freq:
                score += word_freq[word]
                word_count += 1
        
        # Normalize by sentence length to avoid bias towards longer sentences
        if word_count > 0:
            sentence_scores[i] = score / word_count
        else:
            sentence_scores[i] = 0
    
    return sentence_scores

def get_sentence_position_score(index, total_sentences):
    """Give higher scores to sentences at beginning and end"""
    if total_sentences <= 1:
        return 1.0
    
    # First and last sentences are often important
    if index == 0:
        return 1.0
    elif index == total_sentences - 1:
        return 0.8
    elif index < total_sentences * 0.2:
        return 0.9
    else:
        return 0.5

def summarize_text(text, num_sentences=3):
    """
    Summarize text using NLP techniques:
    1. TF-IDF scoring
    2. Word frequency analysis
    3. Sentence position scoring
    """
    # Preprocess
    text = preprocess_text(text)
    
    if not text:
        return "Please provide some text to summarize."
    
    # Tokenize into sentences
    sentences = tokenize_sentences(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    # Get stopwords
    stopwords = get_stopwords()
    
    # Calculate word frequencies for the entire document
    all_words = tokenize_words(text)
    word_freq = calculate_word_frequencies(all_words, stopwords)
    
    # Calculate IDF values
    idf = calculate_idf(sentences, stopwords)
    
    # Score each sentence using multiple criteria
    final_scores = {}
    
    for i, sentence in enumerate(sentences):
        # TF-IDF score
        tfidf_score = calculate_tfidf_score(sentence, idf, stopwords)
        
        # Word frequency score
        words = tokenize_words(sentence)
        freq_score = sum(word_freq.get(w, 0) for w in words)
        if len(words) > 0:
            freq_score /= len(words)
        
        # Position score
        position_score = get_sentence_position_score(i, len(sentences))
        
        # Sentence length penalty (very short or very long sentences)
        word_count = len(words)
        if word_count < 5:
            length_score = 0.5
        elif word_count > 50:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # Combined score with weights
        final_scores[i] = (
            0.4 * tfidf_score + 
            0.3 * freq_score + 
            0.2 * position_score + 
            0.1 * length_score
        )
    
    # Select top sentences
    ranked_sentences = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, score in ranked_sentences[:num_sentences]])
    
    # Build summary maintaining original order
    summary = ' '.join([sentences[i] for i in top_indices])
    
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    summarization_type = data.get('type', 'extractive')  # Default to extractive
    length_setting = data.get('length', 'medium')  # Default to medium
    
    if not text.strip():
        return jsonify({'error': 'Please provide some text to summarize.'}), 400
    
    if summarization_type == 'abstractive':
        # Set parameters based on length preference
        if length_setting == 'short':
            max_len, min_len = 150, 50
        elif length_setting == 'long':
            max_len, min_len = 400, 150
        else:  # medium
            max_len, min_len = 300, 100
            
        # Use BART for abstractive summarization
        summary = summarize_abstractive(text, max_length=max_len, min_length=min_len)
        
        # Count sentences in original and summary
        original_sentences = len(tokenize_sentences(text))
        summary_sentences = len(tokenize_sentences(summary)) if summary else 0
        
        return jsonify({
            'summary': summary,
            'original_sentences': original_sentences,
            'summary_sentences': summary_sentences,
            'type': 'abstractive'
        })
    
    else:
        # Use extractive summarization (existing method)
        sentences = tokenize_sentences(text)
        
        # Adjust number of sentences based on length preference
        if length_setting == 'short':
            num_sentences = max(2, min(3, len(sentences) // 5))
        elif length_setting == 'long':
            num_sentences = max(5, min(10, len(sentences) // 2))
        else:  # medium
            num_sentences = max(3, min(6, len(sentences) // 3))
        
        summary = summarize_text(text, num_sentences)
        
        return jsonify({
            'summary': summary,
            'original_sentences': len(sentences),
            'summary_sentences': num_sentences,
            'type': 'extractive'
        })


@app.route('/summarize_url', methods=['POST'])
def summarize_url():
    data = request.get_json()
    url = data.get('url', '').strip()
    summarization_type = data.get('type', 'extractive')  # Default to extractive
    length_setting = data.get('length', 'medium')  # Default to medium
    
    if not url:
        return jsonify({'error': 'Please provide a URL.'}), 400
    
    # Add http if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Extract text from URL
    text, error = extract_text_from_url(url)
    
    if error:
        return jsonify({'error': error}), 400
    
    if summarization_type == 'abstractive':
        # Set parameters based on length preference
        if length_setting == 'short':
            max_len, min_len = 150, 50
        elif length_setting == 'long':
            max_len, min_len = 400, 150
        else:  # medium
            max_len, min_len = 300, 100
            
        # Use BART for abstractive summarization
        summary = summarize_abstractive(text, max_length=max_len, min_length=min_len)
        
        # Count sentences in original and summary
        original_sentences = len(tokenize_sentences(text))
        summary_sentences = len(tokenize_sentences(summary)) if summary else 0
        
        return jsonify({
            'summary': summary,
            'original_sentences': original_sentences,
            'summary_sentences': summary_sentences,
            'extracted_text': text[:500] + '...' if len(text) > 500 else text,
            'type': 'abstractive'
        })
    
    else:
        # Use extractive summarization (existing method)
        sentences = tokenize_sentences(text)
        
        # Adjust number of sentences based on length preference
        if length_setting == 'short':
            num_sentences = max(2, min(3, len(sentences) // 5))
        elif length_setting == 'long':
            num_sentences = max(5, min(10, len(sentences) // 2))
        else:  # medium
            num_sentences = max(3, min(6, len(sentences) // 3))
        
        summary = summarize_text(text, num_sentences)
        
        return jsonify({
            'summary': summary,
            'original_sentences': len(sentences),
            'summary_sentences': num_sentences,
            'extracted_text': text[:500] + '...' if len(text) > 500 else text,
            'type': 'extractive'
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
