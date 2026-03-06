"""
Main Flask Application for News Summarizer
Supports both Extractive and Abstractive Summarization
"""
from flask import Flask, render_template, request, jsonify
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Import our custom modules
from extractive import summarize_text, tokenize_sentences
from abstractive import summarize_abstractive

app = Flask(__name__)


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
        if not article_text or len(article_text) < 100:
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


@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle text summarization requests"""
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
    """Handle URL summarization requests"""
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
    print("🚀 Starting News Summarizer...")
    print("📊 Extractive Summarization: TF-IDF Algorithm")
    print("🤖 Abstractive Summarization: BART Model") 
    print("🌐 Server: http://localhost:5000")
    print("💡 Press Ctrl+C to stop the server")
    app.run(debug=True, port=5000)