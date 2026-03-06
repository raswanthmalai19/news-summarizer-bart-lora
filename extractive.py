"""
Extractive Summarization using TF-IDF and Sentence Scoring
"""
import re #Cleaning text
import math
from collections import Counter


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
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    word_freq = Counter(filtered_words)
    
    # Normalize frequencies
    max_freq = max(word_freq.values()) if word_freq else 1
    normalized_freq = {word: freq / max_freq for word, freq in word_freq.items()}
    
    return normalized_freq


def calculate_tf(word, document):
    """Calculate Term Frequency for a word in document"""
    return document.count(word) / len(document)


def calculate_idf(sentences, stopwords):
    """Calculate Inverse Document Frequency for all words"""
    N = len(sentences)
    idf_dict = {}
    all_words = set()
    
    # Get all words from all sentences
    for sentence in sentences:
        words = tokenize_words(sentence)
        all_words.update(words)
    
    for word in all_words:
        if word not in stopwords and len(word) > 2:
            containing_sentences = sum(1 for sentence in sentences if word in tokenize_words(sentence))
            if containing_sentences > 0:
                idf_dict[word] = math.log(N / containing_sentences)
    
    return idf_dict


def calculate_tfidf_score(sentence, idf, stopwords):
    """Calculate TF-IDF score for a sentence"""
    words = tokenize_words(sentence)
    tfidf_score = 0
    
    if len(words) == 0:
        return 0
    
    for word in words:
        if word not in stopwords and word in idf:
            tf = calculate_tf(word, words)
            tfidf_score += tf * idf[word]
    
    return tfidf_score / len(words)


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